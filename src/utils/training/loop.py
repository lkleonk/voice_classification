from typing import Optional

import mlflow
import numpy as np
import optuna
import torch
from optuna.trial import Trial
from sklearn.metrics import balanced_accuracy_score

from src.schemas.dataclasses import TrainingArtefactsManager
from src.settings import TRAINING
from src.utils.reporting.mlflow_helper import safe_mlflow_call
from src.utils.training import model_manager
from src.utils.toolkit.auroc import calculate_dynamic_auroc


class DecayLR:       
    def __init__(self, epochs, offset, decay_epochs):
        # Ensure that decay starts before training ends
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "training epochs < decay not allowed"

        self.epochs = epochs              # Total number of training epochs
        self.offset = offset              # Optional epoch shift (usually 0)
        self.decay_epochs = decay_epochs  # Epoch at which decay starts

    def step(self, epoch):
        """
        Returns a learning rate multiplier based on the current epoch.
        The multiplier linearly decays from 1.0 to 0.0 starting at `decay_epochs` until `epochs`.
        """
        # Compute how far past the decay start we are (if at all)
        decay_progress = max(0, epoch + self.offset - self.decay_epochs)

        # Normalize progress over the decay range
        decay_ratio = decay_progress / (self.epochs - self.decay_epochs)

        # Subtract from 1.0 to get the multiplier (decays to 0.0)
        return 1.0 - decay_ratio






# for creating the table


def training_loop(  # give config instead of smth else - criterion stays outside bcs its needed for testing
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    criterion_w_weights,
    train_dataloader,
    val_dataloader,
    device: torch.device,
    logger,
    args,
    add_demographic_data: bool,
    weighted_loss_mode: bool,
    lr_decay_type: str,
    eta_min: float,
    lr_decay_mode: bool = False,
    mlflow_enabled: bool = True,
    trial: Optional[Trial] = None,        # optuna trial
) -> TrainingArtefactsManager:

    # --- INITIALIZATIONS ---
    lrs = []
    epoch = 0
    lr_scheduler = None
    validation_bal_accuracy = 0.0
    model_m = model_manager.ModelManager()

   
    if lr_decay_mode:
        
        if lr_decay_type == 'custom_lr':
            lr_lambda = DecayLR(args.epochs, 0, 0).step
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        elif lr_decay_type == 'cosine_annealing_w_lr_decay':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,      # total epochs until reaching eta_min
                eta_min=eta_min   # minimum learning rate
            )
        else:
            raise ValueError(f"Unknown lr_decay_type: {lr_decay_type}")


    train_losses = [] 
    val_losses = []

    weighted_train_losses = []
    weighted_val_losses = []
    
    val_aurocs = []
    val_balanced_accuracies = []

    epochs = args.epochs
    for epoch in range(epochs):
        logger.info(f'epoch nr. {epoch+1}')
        model.train()

        total_loss_per_epoch = 0.0
        total_weighted_loss_per_epoch = 0.0
        total_samples = 0

        for batch_idx, (xb, yb, demo_b) in enumerate(train_dataloader):
            xb, yb, demo_b = xb.to(device), yb.to(device), demo_b.to(device)

            optimizer.zero_grad()

            if add_demographic_data:
                preds = model(xb, demo_b)
            else:
                preds = model(xb)

            loss = criterion(preds, yb)

            



            if weighted_loss_mode: 
                loss_w_weights = criterion_w_weights(preds, yb)

                back_loss = loss_w_weights

                total_weighted_loss_per_epoch += loss_w_weights.item() * xb.size(0)

            else:
                back_loss = loss


            back_loss.backward()
            optimizer.step()
            #total_loss_per_epoch += loss.item() * xb.size(0)
            total_loss_per_epoch += loss.item() * xb.size(0)  # .item() converts to Python float
            total_samples += xb.size(0)



        
        avg_train_loss_per_sample = total_loss_per_epoch / total_samples
        avg_weighted_train_loss_per_sample = total_weighted_loss_per_epoch / total_samples


        # Validation loss
        model.eval()
        total_weighted_val_loss = 0.0
        total_val_loss = 0.0
        total_val_samples = 0



        validation_all_preds = []
        validation_all_targets = []
        validation_all_probs = []

        with torch.no_grad():
            for batch_idx, (xb, yb, demo_b) in enumerate(val_dataloader):
                xb, yb, demo_b = xb.to(device), yb.to(device), demo_b.to(device)

                # Forward pass
                if add_demographic_data:
                    preds = model(xb, demo_b)
                else:
                    preds = model(xb)

                # Compute loss
                weighted_val_loss = criterion_w_weights(preds, yb)
                val_loss = criterion(preds, yb)
                # Accumulate losses
                total_weighted_val_loss += weighted_val_loss.item() * xb.size(0)
                total_val_loss += val_loss.item() * xb.size(0)
                total_val_samples += xb.size(0)

                # store probabilities for AUROC
                probs = torch.softmax(preds, dim=1)
                validation_all_probs.extend(probs.detach().cpu().numpy())

                # Predictions: argmax over class dimension
                validation_pred_labels = preds.argmax(dim=1).cpu().numpy()
                validation_all_preds.extend(validation_pred_labels)
                validation_all_targets.extend(yb.cpu().numpy())

        
        if lr_decay_mode and lr_scheduler: # valid for both 'custom_lr' and 'cosine_annealing_w_lr_decay'
            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]  # single LR value
            #logger.info(f'Last learning rate value: {current_lr:.6f}')
            lrs.append(current_lr)
        else:
            current_lr = optimizer.param_groups[0]["lr"]


        avg_weighted_val_loss = total_weighted_val_loss / total_val_samples
        avg_val_loss_per_sample = total_val_loss / total_val_samples


        # get balanced accuracy - calculated again for every epoch
        validation_bal_accuracy = balanced_accuracy_score(validation_all_targets, validation_all_preds) 
        validation_accuracy = np.mean(np.array(validation_all_targets) == np.array(validation_all_preds))

        # AUROC handling using centralized utility
        val_auroc_macro_ovr = calculate_dynamic_auroc(
            np.array(validation_all_targets),
            np.array(validation_all_probs),
            logger,
            info=f"(Epoch {epoch+1} Val)"
        )
        
        # update policy (min epoch / test-run override) is owned by ModelManager
        model_m.consider_updating_best_model(
            new_model=model,
            new_mba_value=validation_bal_accuracy,
            new_auroc_value=val_auroc_macro_ovr,
            epoch=epoch,
            logger=logger,
            min_epochs=TRAINING.EARLY_STOP_MIN_EPOCHS,
            test_run=args.test_run,
        )


        
        # --- 1. Update History Lists ---
        train_losses.append(avg_train_loss_per_sample)
        val_losses.append(avg_val_loss_per_sample)
        weighted_train_losses.append(avg_weighted_train_loss_per_sample)
        weighted_val_losses.append(avg_weighted_val_loss)
        val_balanced_accuracies.append(validation_bal_accuracy)
        val_aurocs.append(val_auroc_macro_ovr)


        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss: {avg_train_loss_per_sample:.4f}; "
            f"Weighted train Loss: {avg_weighted_train_loss_per_sample:.4f}; "
            f"Val Loss: {avg_val_loss_per_sample:.4f}; "
            f"Weighted Val Loss: {avg_weighted_val_loss:.4f}; "
            f"Val Balanced Acc: {validation_bal_accuracy:.4f}; "
            f"Val Non-balanced Acc: {validation_accuracy:.4f}; "
            f"Val AUROC (macro-ovr): {val_auroc_macro_ovr:.4f}"
        )

        mlflow_metrics = {
            "train_loss": avg_train_loss_per_sample,
            "weighted_train_loss": avg_weighted_train_loss_per_sample,
            "val_loss": avg_val_loss_per_sample,
            "weighted_val_loss": avg_weighted_val_loss,
            "val_balanced_accuracy": validation_bal_accuracy,
            "val_accuracy": validation_accuracy,
            "val_auroc_macro_ovr": val_auroc_macro_ovr,  
            "current_lr": current_lr,
        }
        mlflow_enabled, _ = safe_mlflow_call(
            mlflow_enabled,
            logger,
            f"log epoch metrics step={epoch+1}",
            mlflow.log_metrics,
            mlflow_metrics,
            step=epoch + 1,
        )
        log_msg = f"Epoch {epoch+1}: " + "; ".join([f"{k}: {v:.4f}" for k, v in mlflow_metrics.items()])
        logger.info(log_msg)


        # Early stopping
        if model_m.should_stop_early(
            current_epoch=epoch,
            patience=TRAINING.EARLY_STOP_PATIENCE,
            min_epochs=TRAINING.EARLY_STOP_MIN_EPOCHS,
        ):
            logger.info(
                "Early stopping triggered at epoch %d: no best-model update for %d epochs.",
                epoch + 1,
                TRAINING.EARLY_STOP_PATIENCE,
            )
            
            break


        # Optuna trial reporting
        if trial:
            #trial.report(avg_val_loss_per_sample, epoch)
            metric = validation_bal_accuracy
            trial.report(metric, epoch)

            # Prune trial if needed
            if trial.should_prune():
                logger.info('-- TRIAL WILL BE PRUNED --')
                raise optuna.TrialPruned()

    if args.test_run:
        logger.info(f'\nepoch {epoch+1} - TEST_RUN: avg train loss list: {train_losses}\navg val loss list: {val_losses}')
    

    # --- 4. Pack the Dataclass ---
    artefacts = TrainingArtefactsManager(
        model_m = model_m,
        train_losses=train_losses,
        val_losses=val_losses,
        weighted_train_losses=weighted_train_losses,
        weighted_val_losses=weighted_val_losses,
        val_balanced_accuracies=val_balanced_accuracies,
        val_aurocs=val_aurocs,
        lrs=lrs,
    )

    return artefacts


