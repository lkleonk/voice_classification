import io
import os
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import confusion_matrix

import src.utils.toolkit.cuda_handling as cuda_handling
import src.utils.toolkit.time_utils as time_utils

device = cuda_handling.set_cuda_to_gpu_nr()



def toPil(plt_figure: Figure) -> Image.Image:    # graph_visualization.py
    """Convert a matplotlib figure to a PIL Image.
    
    Args:
        plt_figure (plt.Figure): A matplotlib figure object.
    
    Returns:
        Image.Image: A PIL Image object.
    """
    # Save the figure to a temporary bytes buffer
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)  # Rewind the buffer to read from the start
    
    # Convert to PIL Image
    pil_img = Image.open(buf)
    
    # Close the figure to free memory
    plt.close(plt_figure)
    
    return pil_img


def convert_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        return [convert_to_cpu(item) for item in data]
    return data



def convert_to_image(sample_data):
    """Convert sample tensor to PIL Image for a single sample"""
    # Convert tensor to numpy array
    if isinstance(sample_data, torch.Tensor):
        sample_np = sample_data.cpu().numpy()
    else:
        sample_np = sample_data
    
    # Handle different tensor shapes
    if len(sample_np.shape) == 4:  # [batch, channel, height, width]
        sample_np = sample_np[0]  # Take first batch
    if len(sample_np.shape) == 3:  # [channel, height, width]
        sample_np = sample_np[0]  # Take first channel (grayscale)
    
    # Normalize to 0-255 range
    sample_np = (sample_np - sample_np.min()) / (sample_np.max() - sample_np.min() + 1e-8) * 255
    sample_np = sample_np.astype(np.uint8)
    
    # Convert to PIL Image
    return Image.fromarray(sample_np)





def get_rec_error_list(model, iterable_dataset, loss, return_sample = False):
    model.to(device)
    loss_list=[]
    x = None
    pred = None

    with torch.no_grad():  # No need to compute gradients
        for i, (x, y, demo) in enumerate(iterable_dataset):
            x = x.unsqueeze(1)
            x = x.to(device)
            pred=model(x)
            #if i == 0:  # Print only for first batch to avoid clutter
                #print(f"Input shape: {x.shape}")
                #print(f"Input range: [{x.min():.8f}, {x.max():.8f}]")  # Add this
                #print(f"Output shape: {pred.shape}")
                #print(f"Output range: [{pred.min():.8f}, {pred.max():.8f}]")  # Add this
            rec_error = loss(pred, x)
            loss_list.append(rec_error.item())        


    
                
    if return_sample:
        assert x is not None and pred is not None, "iterable_dataset was empty"

        #first sample of x and pred:
        input = convert_to_image(x.squeeze())
        output = convert_to_image(pred.squeeze())
        diff = convert_to_image(x-pred)


        return loss_list, input, output, diff
    else:
        return loss_list, None, None, None



def get_rec_error_distr(model, train_dataset, val_dataset, test_dataset, control_dataset, loss, additional_info='', return_sample_imgs=False):
    spec_input, spec_output, spec_diff = None, None, None

    train_list, _, _, _ = get_rec_error_list(model, train_dataset, loss)
    val_list, _, _, _ = get_rec_error_list(model, val_dataset, loss)
    test_list, _, _, _ = get_rec_error_list(model, test_dataset, loss)
    if return_sample_imgs:
        control_list, spec_input, spec_output, spec_diff = get_rec_error_list(model, control_dataset, loss, return_sample = True)
    else:
        control_list, _, _, _ = get_rec_error_list(model, control_dataset, loss)

    mean_stats = (
        f"Train   - mean: {np.mean(train_list):.4f}, std: {np.std(train_list):.4f}\n"
        f"Val     - mean: {np.mean(val_list):.4f}, std: {np.std(val_list):.4f}\n"
        f"Test    - mean: {np.mean(test_list):.4f}, std: {np.std(test_list):.4f}\n"
        f"Control - mean: {np.mean(control_list):.4f}, std: {np.std(control_list):.4f}"
    )


    # get max value from train + val + test list
    max_benchmark_value = np.max(np.concatenate([train_list, val_list, test_list]))
    # how much values in control_list below max_benchmark_value: abs number + %
    # how much values in control_list over max_benchmark_value: abs number + %

    threshold_stats = get_threshold_above_and_below_stats(max_benchmark_value, control_list)
    stats = f'{additional_info}\n{mean_stats}\n{threshold_stats}'

    pil_img = plot_distribution(train_list, val_list, test_list, control_list, title = f'Reconstruction Error Distribution{additional_info}')

    return pil_img, stats, spec_input, spec_output, spec_diff

    
def get_threshold_above_and_below_stats(bench, rec_error_list):
    # Calculate control vs benchmark statistics
    control_errors = np.array(rec_error_list)
    below_threshold = np.sum(control_errors <= bench)
    above_threshold = np.sum(control_errors > bench)
    total_control = len(control_errors)
    
    below_percent = (below_threshold / total_control) * 100
    above_percent = (above_threshold / total_control) * 100
    
    stats_msg = (
            f"\nControl vs Benchmark Statistics:\n"
            f"Max benchmark value: {bench:.4f}\n"
            f"Control samples below threshold: {below_threshold}/{total_control} ({below_percent:.2f}%)\n"
            f"Control samples above threshold: {above_threshold}/{total_control} ({above_percent:.2f}%)"
        )
    return stats_msg


def plot_distribution(train_distribution, val_distribution, test_distribution, anomalous_distribution, saving_path = '', color1='blue', color2='green', color3='cyan', anomalous_color='red', title='Reconstruction Error Distribution', option='return', x_label = 'Reconstruction Error'):
    # Clear the entire figure, including axes and histograms
    fig = plt.figure()

    if not isinstance(train_distribution, (list, np.ndarray)):
        raise TypeError("ERROR: train_distribution is not a list or a np.array")
    
    # Create a histogram of normal recerrors
    plt.hist(train_distribution, bins=50, color=color1, alpha=0.5, label='training images - copd')
    plt.hist(val_distribution, bins=50, color=color2, alpha=0.5, label='validation images - copd')
    plt.hist(test_distribution, bins=50, color=color3, alpha=0.5, label='test images - copd')
    plt.hist(anomalous_distribution, bins=50, color=anomalous_color, alpha=0.5, label='Anomalous Images - control')
      

    # Set labels and legend
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()

    # choose option
    if option == 'save':
        if saving_path == '':
            print('error: no saving path given for plot')
            plt.close(fig)
        else:
            plt.savefig(saving_path)
            plt.close(fig)
            print('plot saved in ' + saving_path)
    elif option == 'return':
        plot_img = toPil(fig)
        plt.close(fig)
        print('Distribution plot returned')
        return plot_img
    elif option == 'show':
        plt.show(fig)
        plt.close(fig)
    else:
        plt.close(fig)
        print('option chosen not known; choose from "save", "return", "show"')







def plot_confidence_distribution(output_dir, logits_list, all_targets, filename="confidence_distribution.png", return_plot=True):
    """
    Plots boxplots of confidence (softmax probability of predicted class) for correct vs wrong predictions,
    with every confidence value shown as a dot, and saves the figure.
    
    Args:
        output_dir (str): Directory to save the figure
        logits_list (list[torch.Tensor]): List of model logits from evaluation batches
        all_targets (list or np.ndarray): Ground truth labels
        filename (str): Name of the saved figure file
    """
    # Concatenate logits
    logits = torch.cat(logits_list, dim=0)

    # Softmax probabilities
    probs = torch.softmax(logits, dim=1)

    # Predicted class and confidence
    pred_confidences, pred_classes = torch.max(probs, dim=1)

    # Convert targets to tensor
    targets = torch.tensor(all_targets, device=logits.device)

    # Masks
    correct_mask = pred_classes == targets
    wrong_mask = ~correct_mask

    # Split confidence values
    correct_confidences = pred_confidences[correct_mask].cpu().numpy()
    wrong_confidences   = pred_confidences[wrong_mask].cpu().numpy()

    # Prepare figure
    fig = plt.figure(figsize=(6,6))
    data = [wrong_confidences, correct_confidences]
    labels = ['Wrong', 'Correct']

    # Boxplot
    plt.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)

    # Add dots (jittered scatter)
    for i, y in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(y))  # jitter
        plt.scatter(x, y, alpha=0.6, color='black', s=10)

    plt.ylabel("Confidence (softmax probability of predicted class)")
    plt.title("Confidence Distribution: Correct vs Wrong Predictions")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    
    #if return_plot:
    pil_img = toPil(fig)
    plt.close(fig)
    return pil_img




def plot_confidence_distribution_old(output_dir, logits_list, all_targets, filename="confidence_distribution.png", return_plot=True):
    """
    Plots and saves the confidence distribution (softmax prob of predicted class)
    for correct vs. incorrect predictions.

    Args:
        output_dir (str): Directory to save the plot
        logits_list (list[torch.Tensor]): List of model logits from evaluation batches
        all_targets (list or np.ndarray): Ground truth labels
        filename (str): Name of the output file (default: 'confidence_distribution.png')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Concatenate logits
    logits = torch.cat(logits_list, dim=0)  # shape [N, num_classes]

    # Softmax probabilities
    probs = torch.softmax(logits, dim=1)

    # Confidence for predicted class
    pred_confidences, pred_classes = torch.max(probs, dim=1)

    # Convert targets to tensor
    targets = torch.tensor(all_targets, device=logits.device)

    # Masks
    correct_mask = pred_classes == targets
    wrong_mask = ~correct_mask

    # Split confidence values
    correct_confidences = pred_confidences[correct_mask].cpu().numpy()
    wrong_confidences   = pred_confidences[wrong_mask].cpu().numpy()

    # Plot
    fig = plt.figure(figsize=(8,6))
    plt.hist(correct_confidences, bins=30, alpha=0.6, label="Correct", density=True)
    plt.hist(wrong_confidences, bins=30, alpha=0.6, label="Wrong", density=True)
    plt.xlabel("Confidence (softmax probability of predicted class)")
    plt.ylabel("Density")
    plt.title("Confidence Distribution: Correct vs Wrong Predictions")
    plt.legend()

    # Save figure
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    if return_plot:
        pil_img = toPil(fig)
        plt.close(fig)
        return pil_img




def plot_lrs(lrs, output_dir = None, filename="lrs.png", return_plot=True):
    """
    Plots learning rates

    Args:
        output_dir (str): Directory to save the plot
        lrs: learning rate list
        filename (str): Name of the output file (default: 'confidence_distribution.png')
        return-plot(bool) whether to return the plot as pillow image
    """
    assert filename or return_plot, 'Either filename or return_plot=True has to be given to function - if not, nothing happens.'

    # Plot the LRs over iterations
    fig, ax = plt.subplots()
    ax.plot(lrs)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Learning Rate")


    # Save figure
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    if return_plot:
        pil_img = toPil(fig)
        plt.close(fig)
        return pil_img





def visualize_loss_curve(file_name, train_loss, val_loss, output_dir = None, title=None, # graph_visualization.py
                        save_in_downloads=False, return_plot=False, show_plot=False):
    """
    Parameters:
        show_plot (bool): If True, displays the plot with plt.show(). Default False.
    """
    if len(train_loss) != len(val_loss):
        raise ValueError("Lengths of train_loss and val_loss must be the same")

    # Convert to CPU if tensors
    train_loss = convert_to_cpu(train_loss)
    val_loss = convert_to_cpu(val_loss)

    epochs = range(1, len(train_loss) + 1)
    title = title or 'Training and Validation Loss'

    fig = plt.figure()
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    # Save to output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{file_name}.png")
        plt.savefig(plot_path)
        print(f"Plot saved in: {plot_path}")

    # Optional save to Downloads
    if save_in_downloads:
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        current_time = time_utils.get_day_month_year_hour_minute_second()
        download_path = os.path.join(downloads_path, f"{file_name}_{current_time}.png")
        plt.savefig(download_path)
        print(f"Plot also saved in: {download_path}")

    # Control plot display
    if show_plot:
        plt.show()
    
    if return_plot:
        pil_img = toPil(fig)
        plt.close(fig)
        return pil_img
    else:
        plt.close(fig)




def plot_confusion_matrix_and_return(y_true, y_pred, labels, output_path=None): # graph_visualization.py
    """
    Plots a confusion matrix and optionally saves it to file, returns as PIL Image.
    
    Parameters:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        labels (list): List of class labels
        output_path (str, optional): If provided, saves the image to this path
        
    Returns:
        PIL.Image: The confusion matrix as a PIL image
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    # Save to file if output path provided
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight')
    
    # Save to buffer and convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf)
    
    return pil_img







def save_spectrogram_sample_as_png(output_dir, spectrogram, label=None, cmap='magma', dpi=100):   # stays here
    """
    Saves a spectrogram (assumed to be 2D or 3D with shape HxWx1) as a PNG image and returns it as PIL Image.

    Parameters:
        output_dir (str): Directory to save the image.
        spectrogram (np.ndarray): Spectrogram array, shape (H, W) or (H, W, 1).
        label (str or int, optional): Optional label for the filename.
        cmap (str): Colormap used for saving the image.

    Returns:
        PIL.Image: The spectrogram as a PIL image
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Remove last dimension if it's (H, W, 1)
    if spectrogram.ndim == 3 and spectrogram.shape[-1] == 1:
        spectrogram = spectrogram[:, :, 0]

    # Set filename
    filename = "spectrogram"
    if label is not None:
        filename += f"_{label}"
    filename += ".png"

    filepath = os.path.join(output_dir, filename)

    height, width = spectrogram.shape
    # Compute figure size in inches based on spectrogram shape and dpi
    figsize = (width / dpi, height / dpi)

    # Create figure with correct size
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('off')
    plt.imshow(spectrogram, cmap=cmap, aspect='auto', origin='lower')
    
    # Save to file
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    
    # Save to buffer and convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf)
    
    return pil_img


 





# TODO normalize colors
def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()


def turn_demo_info_into_graph(demographic_category_list, table):
    """
    Generate plots (boxplots for numeric, bar plots for categorical)
    for a list of demographic categories and return as PIL Images.
    """
    pil_images = []

    for tag in demographic_category_list:
        if tag == 'sex':
            images = turn_into_barchart([tag], table)
        else:
            images = turn_into_boxplots([tag], table)
        pil_images.extend(images)

    return pil_images


def turn_into_boxplots(dem_list, table):
    """
    Generate boxplots for numeric columns by 'correct' column.
    Returns a list of PIL Images.
    """
    pil_images = []

    for tag in dem_list:
        temp_table = table.copy()
        temp_table['correct'] = temp_table['correct'].astype('category')

        # Convert numeric and drop NaNs
        temp_table[tag] = pd.to_numeric(temp_table[tag], errors='coerce')
        temp_table = temp_table.dropna(subset=[tag])
        if temp_table.empty:
            continue

        # Create boxplot
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x='correct', y=tag, data=temp_table, ax=ax)
        ax.set_title(f'{tag} Distribution by Prediction Correctness')
        ax.set_xlabel('Prediction Correct (True/False)')
        ax.set_ylabel(tag)

        pil_images.append(_fig_to_pil(fig))

    return pil_images


def turn_into_barchart(dem_list, table):
    """
    Generate bar/count plots for categorical columns by 'correct' column.
    Returns a list of PIL Images.
    """
    pil_images = []

    for tag in dem_list:
        temp_table = table.copy()
        temp_table[tag] = temp_table[tag].astype('category')
        temp_table['correct'] = temp_table['correct'].astype('category')

        if temp_table.empty:
            continue

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='correct', hue=tag, data=temp_table, ax=ax)
        ax.set_title(f'{tag} Distribution by Prediction Correctness')
        ax.set_xlabel('Prediction Correct (True/False)')
        ax.set_ylabel('Count')
        ax.legend(title=tag)

        pil_images.append(_fig_to_pil(fig))

    return pil_images

def get_category_chart(tag, train_values, val_values, test_values):
    """
    Creates a bar chart for categorical demographic values (e.g., 'sex').
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6, 4))

    # Count frequencies
    train_counts = Counter(train_values)
    val_counts = Counter(val_values)
    test_counts = Counter(test_values)

    # All categories across splits
    categories = sorted(set(train_counts) | set(val_counts) | set(test_counts))

    train = [train_counts[c] for c in categories]
    val   = [val_counts[c] for c in categories]
    test  = [test_counts[c] for c in categories]

    x = np.arange(len(categories))
    width = 0.25

    # Plot using the axes
    ax.bar(x - width, train, width, label='Train')
    ax.bar(x,         val,   width, label='Validation')
    ax.bar(x + width, test,  width, label='Test')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count")
    ax.set_title(f"Category Distribution for '{tag}'")
    ax.legend()

    fig.tight_layout()

    return _fig_to_pil(fig)




def get_distribution_chart(tag, train_values, val_values, test_values):
    """
    Creates an overlaid histogram for numerical demographic values (e.g., 'age').
    None values are ignored.
    """
    # Filter out None values
    train_values = [v for v in train_values if v is not None]
    val_values   = [v for v in val_values if v is not None]
    test_values  = [v for v in test_values if v is not None]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6, 4))

    bins = 20  # reasonable default

    # Histograms
    ax.hist(train_values, bins=bins, alpha=0.5, label='Train')
    ax.hist(val_values,   bins=bins, alpha=0.5, label='Validation')
    ax.hist(test_values,  bins=bins, alpha=0.5, label='Test')

    ax.set_xlabel(tag)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of '{tag}'")
    ax.legend()

    fig.tight_layout()

    return _fig_to_pil(fig)



def _fig_to_pil(fig):
    """
    Helper function to convert a Matplotlib figure to a PIL Image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    plt.close(fig)
    buf.close()
    return pil_img



def get_boxplots_for_demographic_data(table):
    try:
        demo_list = ['age', 'sex', 'height_cm', 'weight_kg', 'bmi']
        pil_images = turn_demo_info_into_graph(demo_list, table)
    except Exception as e:
        print("get_boxplots_for_demographic_data: Falling back due to error (no height_cm etc in data):", e)
        demo_list = ['age', 'sex', 'age']
        pil_images = turn_demo_info_into_graph(demo_list, table)

    return pil_images  # list of PIL.Image objects


