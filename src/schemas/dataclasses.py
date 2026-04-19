from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL.Image import Image
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    import optuna

    from src.schemas.enums import RunMode
    from src.utils.training.model_manager import ModelManager
"""
DATACLASSES, LISTS, AND MUTABLE STATE Ã¢â‚¬â€ SUMMARY

1) The core problem
- Using mutable defaults like [] in dataclasses causes SHARED STATE across instances.
- Using typing.List as a default is invalid because List is not callable.

2) default_factory=list
- field(default_factory=list) creates a NEW list per instance.
- This avoids shared mutable state.
- Side effect: the field becomes OPTIONAL in __init__ (can be omitted).
- It does NOT make the field Optional[T]; it will never be None.

3) Optional vs default
- Optional[List[T]] = None  -> value may be None (semantic absence).
- List[T] = field(default_factory=list) -> always a list, possibly empty.
- These solve different problems and should not be confused.

4) If you want NO shared state AND the field MUST be passed
- Do NOT use default_factory.
- Make the field required and defensively copy it in __post_init__:

  xs: List[T]
  __post_init__: self.xs = list(self.xs)

- This guarantees:
  - caller must pass xs
  - each instance owns its own list
  - even reused input lists wonÃ¢â‚¬â„¢t be shared

5) deepcopy discussion
- deepcopy at call sites works but is discouraged:
  - slow
  - overkill
  - relies on caller discipline
- Better: copy INSIDE the dataclass.
- Prefer shallow copies (list(), dict()).
- Use deepcopy only for deeply nested structures where isolation is required.

6) Design rules of thumb
- Mutable + optional + empty OK Ã¢â€ â€™ default_factory=list
- Mutable + required Ã¢â€ â€™ no default + copy in __post_init__
- Missing vs empty matters Ã¢â€ â€™ Optional[List[T]] = None
- frozen=True freezes the attribute reference, NOT the list contents.

7) One-line takeaway
- default_factory=list = fresh list + optional constructor argument
- Required + safe = accept the list and copy it inside the dataclass
"""


@dataclass(frozen=True) #, kw_only=True)
class TrainingResult:
    mode: "RunMode"

    # Metrics (required for downstream CV)
    final_model_val_mba: Optional[float] = None
    final_model_test_mba: Optional[float] = None
    final_model_val_auroc: Optional[float] = None
    final_model_test_auroc: Optional[float] = None

    best_model_val_mba: Optional[float] = None
    best_model_test_mba: Optional[float] = None
    best_model_val_auroc: Optional[float] = None
    best_model_test_auroc: Optional[float] = None

    # Optional artifacts / info
    pil_images: Optional[List[Image]] = None
    report_info: Optional[str] = None


@dataclass(frozen=True)
class SavedModelInfo:
    configuration: Dict[str, Any]
    arguments: Dict[str, Any]
    test_report_info: Any
    schema_version: str = "1.0"
    model_label: Optional[str] = None
    epoch_index: Optional[int] = None
    epoch_human: Optional[int] = None
    final_val_loss: Optional[float] = None
    final_test_loss: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra")
        data.update(extra)
        return data

    def save_yaml(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.to_dict(), file, sort_keys=False)
        return path


@dataclass(frozen=True)
class TrainingArtefactsManager:
    # --- Primary Metrics (Lists for History) ---
    # for TRAINING, not TESTING
    model_m: ModelManager
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)

    # Explicit metrics you care about
    val_balanced_accuracies: List[float] = field(default_factory=list)
    val_aurocs: List[float] = field(default_factory=list)

    # Optional / secondary
    weighted_train_losses: List[float] = field(default_factory=list)
    weighted_val_losses: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)

    def get_final_metrics(self) -> Dict[str, float]:
        """
        Returns the specific snapshot needed for CrossValSummary. FROM TRAINING (NOT TESTING)
        """
        return {
            "val_mba": self.val_balanced_accuracies[-1] if self.val_balanced_accuracies else 0.0,
            "val_auroc": self.val_aurocs[-1] if self.val_aurocs else 0.0,
        }

    def get_last_val_balanced_accuracy(self) -> float:
        """
        Returns the most recent validation balanced accuracy.
        Safe default if none is recorded yet.
        """
        return self.val_balanced_accuracies[-1] if self.val_balanced_accuracies else 0.0


    def get_last_auroc(self) -> float:
        """
        Returns the most recent validation balanced accuracy.
        Safe default if none is recorded yet.
        """
        return self.val_aurocs[-1] if self.val_aurocs else 0.0


@dataclass(frozen=True)
class TrainingSetup:
    # data / labels
    dataset_manager: Any

    # loaders
    train_dataloader: DataLoader
    val_dataloader: DataLoader

    # model + training objects
    model: torch.nn.Module
    criterion: nn.Module
    criterion_w_weights: Optional[nn.Module]
    optimizer: torch.optim.Optimizer
    model_m: Any

    # meta
    num_classes: int
    img_height: int
    img_width: int


@dataclass() # kw_only=True)
class SpecSample:
    spec: np.ndarray
    label: int
    index: int
    pil_img: Optional[Image] = None
    pil_gradcam_img_by_label: Optional[Image] = None
    pred: Optional[int] = None

    def sample_label_index(
        self
    ) -> Tuple[np.ndarray, int, int, Optional[Image], Optional[int]]:
        """
        Return (spec, label, index, pil_img, pred) as a tuple.
        """
        return self.spec, self.label, self.index, self.pil_img, self.pred

    def __str__(self) -> str:
        """Human-friendly description."""
        base = f"Spectrogram sample #{self.index} with label {self.label}"
        if self.pred is not None:
            base += f" (pred={self.pred})"
        return base

    def __repr__(self) -> str:
        """Developer-friendly description."""
        pil_img_status = "Exists" if self.pil_img is not None else "None"
        pil_gradcam_img_status = (
            "Exists" if self.pil_gradcam_img_by_label is not None else "None"
        )
        return (
            f"\nSpecSample(\n"
            f"  label={self.label},\n"
            f"  index={self.index},\n"
            f"  pred={self.pred},\n"
            f"  spec_shape={self.spec.shape},\n"
            f"  pil_img={pil_img_status},\n"
            f"  pil_gradcam_img={pil_gradcam_img_status}\n"
            f")"
        )


AddDemoStatus = Literal["present", "missing_key", "no_config"]


@dataclass(frozen=True) # , kw_only=True)
class SheetResult:
    # identity (used for grouping/pivoting)
    dataset: str                 # e.g. "uk"
    lung_conditions: str         # e.g. "copd, control"
    recording_category: str      # e.g. "a", "o", "i", "poem"
    model: str                   # e.g. "pann", "cnnlstm"

    # metrics (means from cross-validation)
    final_model_val_mba_mean: float
    final_model_test_mba_mean: float
    final_model_val_auroc_mean: float
    final_model_test_auroc_mean: float

    # stability numbers
    final_model_val_mba_std: float
    final_model_test_mba_std: float
    final_model_val_auroc_std: float
    final_model_test_auroc_std: float

    # metrics (means from cross-validation)
    best_model_val_mba_mean: float
    best_model_test_mba_mean: float
    best_model_val_auroc_mean: float
    best_model_test_auroc_mean: float

    # stability numbers
    best_model_val_mba_std: float
    best_model_test_mba_std: float
    best_model_val_auroc_std: float
    best_model_test_auroc_std: float

    norm_mode: str  # e.g. "norm_on" / "norm_off"
    hyperparam_config: Optional[Dict[str, Any]] = None # config used during training
    add_demographic_data_var: Optional[bool] = None # config used during training


    def get_add_demographic_data_info(self) -> Tuple[bool, AddDemoStatus]:
        """
        Returns:
        - value: bool
        - status:
            "present"   -> value explicitly set (trustworthy)
            "no_config" -> value missing / not attached
        """
        if self.add_demographic_data_var is None:
            return False, "no_config"

        return bool(self.add_demographic_data_var), "present"





@dataclass
class OptunaStudyResult:
    study_id: str
    best_trial_number: int
    best_params: Dict[str, Any]
    best_value: float
    best_trial_config_path_list: List[str]
    study: "optuna.study.Study"
    final_training_result_instance: Optional[TrainingResult] = None
    output_dir: str = ""



@dataclass
class EvalResult: 
    info: str

    # core metrics
    avg_loss: float
    accuracy: float
    balanced_accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    auroc_macro_ovr: float  # can be NaN if not computable

    # outputs
    confusion_matrix: np.ndarray
    classification_report: str
    metrics_text: str  # pretty string you can logger.info()

    # raw predictions
    targets: np.ndarray
    preds: np.ndarray

    # optional: keep if you need later analysis/plots
    logits: Optional[np.ndarray] = None   # shape (N, C)
    probs: Optional[np.ndarray] = None    # shape (N, C)

