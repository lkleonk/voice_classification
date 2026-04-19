from __future__ import annotations

from enum import Enum, StrEnum
from pathlib import Path
from typing import Final, Sequence

"""
settings.py

Single source of truth for CONSTANTS used across the ML pipeline.

How to use
----------
Import a subsection:
    from settings import PATHS, PRUNING
    PATHS.MODEL_WEIGHTS_DIR
    PRUNING.OPTUNA.MIN_EPOCHS

Design rules (do not violate)
-----------------------------
- Only declarative assignments (constants).
- No environment access, no I/O, no Optuna / MLflow instantiation.
- No side effects, no logic beyond lightweight expressions.
- This file defines *policy*, not runtime behavior.

"""



# =============================================================================
# PATHS
# =============================================================================

class SchemaVersion(StrEnum):
    V0_TEST_RUN = "v0_test_run"
    V1 = "v1"

CURRENT_SCHEMA_VERSION = SchemaVersion.V1


class PATHS:
    """Project-wide filesystem layout (repo-local, deterministic).
    """

    # project/.../settings.py Ã¢â€ â€™ project/
    PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

    # parent/
    PARENT_ROOT: Final[Path] = PROJECT_ROOT.parent

    SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
    WORK_DIR: Final[Path] = PARENT_ROOT / "work" # HPC-safe workspace
    OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"

    # Explicit pretrained weights policy (no ~/.cache usage)
    MODEL_WEIGHTS_DIR: Final[Path] = WORK_DIR / "model_weights"
    # Direct file paths
    PANN_WEIGHTS_PATH: Final[Path] = MODEL_WEIGHTS_DIR / "pann" / "Cnn14_16k_mAP=0.438.pth" # todo: wire these pats to src/spectrogram/model/vggish/ and .../pann/ and .../passt/
    PASST_WEIGHTS_PATH: Final[Path] = MODEL_WEIGHTS_DIR / "passt" / "passt-s-kd-ap.486.pt" # passt_s_kd_p16_128_ap486
    VGGISH_WEIGHTS_PATH: Final[Path] = MODEL_WEIGHTS_DIR / "vggish" / "vggish-10086976.pth"
    EFFICIENTAT_WEIGHTS_DIR: Final[Path] = MODEL_WEIGHTS_DIR / "efficientat"
    EFFICIENTAT_WEIGHTS_PATH: Final[Path] = EFFICIENTAT_WEIGHTS_DIR / "dymn20_as_mAP_493.pt"
    WAV2VEC2_WEIGHTS_DIR: Final[Path] = MODEL_WEIGHTS_DIR / "wav2vec2"
    WAV2VEC2_VARIANT_ID: Final[str] = "facebook/wav2vec2-large-robust"
    BEATS_WEIGHTS_DIR: Final[Path] = MODEL_WEIGHTS_DIR / "beats"
    BEATS_VARIANT: Final[str] = "BEATs_iter3"
    BEATS_FILE_NAME: Final[str] = f"{BEATS_VARIANT}.pt"
    BEATS_WEIGHTS_PATH: Final[Path] = BEATS_WEIGHTS_DIR / BEATS_FILE_NAME
    AST_WEIGHTS_DIR: Final[Path] = MODEL_WEIGHTS_DIR / "ast"
    AST_VARIANT_NAME: Final[str] = "ast-base-audioset"
    AST_VARIANT_DIR: Final[Path] = AST_WEIGHTS_DIR / AST_VARIANT_NAME
    AST_HF_VARIANT_IDENTIFIER: str = "MIT/ast-finetuned-audioset-10-10-0.4593"

    # create_exhaustive_study_sheet.py persists best-trial config YAMLs here.
    BEST_CONFIGS_DIR: Final[Path] = WORK_DIR / "best_configs"
    BEST_CONFIGS_SPECIFIC_DIR: Final[Path] = BEST_CONFIGS_DIR / "specific_configs"
    CURRENT_SPECIFIC_CONFIGS_DIR: Final[Path] = BEST_CONFIGS_SPECIFIC_DIR / "first_round"
    BEST_CONFIGS_DICT_FILENAME_SUFFIX: Final[str] = "_best_trials_configs.yml"

    TARGET_BEST_CONFIG_VERSION_DIR = CURRENT_SCHEMA_VERSION
    LOAD_BEST_CONFIG_VERSION_DIR: Final[str] = TARGET_BEST_CONFIG_VERSION_DIR
    # relevant for folder creation during excel summary

    PICKLED_SHEET_RESULTS_DIR: Final[Path] = WORK_DIR / "pickled_sheet_results"
    SHEET_RESULTS_PICKLE_SUFFIX: Final[str] = "_sheet_results.pkl"
    TARGET_SHEET_RESULTS_DIR: Final[Path] = PICKLED_SHEET_RESULTS_DIR / TARGET_BEST_CONFIG_VERSION_DIR
    LOAD_SHEET_RESULTS_DIR: Final[Path] = PICKLED_SHEET_RESULTS_DIR / TARGET_BEST_CONFIG_VERSION_DIR

    OPTUNA_SUMMARY_DIR: Final[Path] = WORK_DIR / "optuna_summaries"

    OPTUNA_ID: str = "optuna" # TODO rename this to optuna_directory_id and the line below to cross_val_directory_id
    CROSS_VAL_ID: str = "cross_val"
    RSEEDS_ID: str = "rseeds"

# =============================================================================
# CLI CHOICES for cli.py - argparser arguments
# =============================================================================

class CLI_CHOICES:
    """Allowed choices for CLI arguments."""

    MODES: Final[tuple[str, ...]] = ("train", "optuna_tuning", "exhaustive_study", "cross_val", "pickle_to_excel")
    SELECTED_CLASSES: Final[tuple[str, ...]] = ("copd,control",)
    RECORDING_CATEGORY: Final[tuple[str, ...]] = ("poem",)
    DATASET_V: Final[tuple[str, ...]] = ("uk",)
    NORM_MODE: Final[tuple[str, ...]] = ("norm_on", "norm_off")
    MODEL_TYPES: Final[tuple[str, ...]] = (
        "autoencoder",
        "test_cnn",
        "test_cnn2",
        "tabular_mlp",
        "cnn",
        "cnnlstm",
        "lstm",
        "vit",
        "passt",
        "pann",
        "efficientat",
        "ast",
        "wav2vec2",
        "beats",
        "vggish"
    )


# =============================================================================
# EXCEL SUMMARY POLICY for create_exhaustive_study_sheet.py:
# =============================================================================

class EXCEL:
    """Exhaustive study sheet behavior toggles."""

    class TuningMode(Enum):
        NO_TUNING = "no_tuning"
        RUN_OPTUNA_FIRST = "run_optuna_first"
        USE_PRECOMPUTED_BEST_CONFIGS = "use_precomputed_best_configs"

    TUNING_MODE = TuningMode.RUN_OPTUNA_FIRST

    BEST_CONFIG_DICT_SCHEMA_VERSION: Final[int] = 3
    BEST_CONFIG_DICT_SCHEMA_NOTE: Final[list] = [
        (2, "Top-level norm mode, then dataset, then canonical combo path to config lists."),
        (3, "Top-level norm mode, then demo-data key, then dataset, then canonical combo path to config lists."),
    ]

    class TEST_GRID:
        LUNG_CONDITIONS: Final[Sequence[str]] = [
            "copd,control",
        ]
        RECORDING_CATEGORY: Final[Sequence[str]] = ["poem"]
        MODELS: Final[Sequence[str]] = ["test_cnn", "test_cnn2", "cnn"]

    class FULL_GRID:
        LUNG_CONDITIONS: Final[Sequence[str]] = [
            "copd,control",
        ]
        RECORDING_CATEGORY: Final[Sequence[str]] = ["poem"]
        MODELS: Final[Sequence[str]] = ["cnn", "cnnlstm", "pann"] # ["efficientat", "passt", "lstm"] #["cnn", "cnnlstm", "lstm", "passt", "pann"]


# =============================================================================
# CROSS VALIDATION
# =============================================================================

class EVALUATION_PROTOCOL:
    """Cross-validation policy and grid definitions.
    """
    K_FOLDS = 5
    TEST_RUN_K_FOLDS = 2

    # If True, skip k-fold splitting and run repeated trainings with different seeds.
    ONLY_CHANGE_RANDOM_SEEDS = True
    TEST_RUN_RANDOM_SEEDS_RUNS = 2
    RANDOM_SEEDS_RUNS = 8
    EVAL_SEED_POOL = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]



class MULTIPROCESSING:
    """Parallel worker limits for CPU/GPU orchestration."""

    # ProcessPoolExecutor workers used by create_exhaustive_study_sheet.py
    MAX_WORKERS_EXHAUSTIVE_STUDY: Final[int] = 20

    # Upper bound for concurrent GPU-bound jobs (for schedulers/orchestrators).
    MAX_GPU_WORKERS: Final[int] = 1


class DATALOADER:
    """Shared DataLoader policy defaults for training and evaluation."""

    # Conservative worker default that avoids oversubscribing CPU on local runs.
    NUM_WORKERS: Final[int] = 2

    # Evaluation stays sample-wise to keep per-item reporting and shapes predictable.
    EVAL_BATCH_SIZE: Final[int] = 1


class DATA_VALIDATION:
    """Validation tolerances for derived dataset artefacts."""

    # Small width drift is expected from preprocessing/windowing differences.
    SPECTROGRAM_WIDTH_TOLERANCE_PX: Final[int] = 15


# =============================================================================
# MODEL SAVING POLICY
# =============================================================================


class MODELS:
    """Model registry and behavior flags by model family.
    """
    PRETRAINED_MODELS_LIST: Final[Sequence[str]] = ["vggish", "pann", "passt", "ast", "efficientat", "wav2vec2", "beats"]
    OWN_SPECTROGRAM_INPUT_MODELS_LIST: Final[Sequence[str]] = ["pann", "passt", "ast", "efficientat", "wav2vec2", "beats"]
    DEMO_ONLY_NO_AUDIO_MODELS_LIST: Final[Sequence[str]] = ["tabular_mlp"]


class MODEL_SAVE:
    """Checkpointing and performance threshold policy.
    """

    # Behavioral flags
    SAVE_BEST_MODEL: Final[bool] = True

    # Filenames
    BEST_CHECKPOINT_FILENAME: Final[str] = "MODEL_BEST.pt"
    FINAL_CHECKPOINT_FILENAME: Final[str] = "MODEL_FINAL.pt"

    # Model selection thresholds by dataset and class count
    THRESHOLD_MAP_UK: Final[dict[int, float]] = {
        2: 0.9,
    }
    DEFAULT_THRESHOLD: Final[float] = 0.5


# =============================================================================
# TEST / DEBUG DEFAULTS
# =============================================================================

class TEST_RUN:
    """Fast-run defaults for local debugging.
    """

    FOLDS: Final[int] = 2


# =============================================================================
# TRAINING (EARLY STOPPING)
# =============================================================================

class TRAINING:
    """Training-time defaults (early stopping, epochs).
    """

    EARLY_STOP_PATIENCE: Final[int] = 5 # probably implement it in snakefile?
    EARLY_STOP_MIN_EPOCHS: Final[int] = 12


# =============================================================================
# MLFLOW
# =============================================================================

class MLFLOW:
    """MLflow naming policy.
    """
    EXPERIMENT_NAME: Final[str] = "voice-classification"


# =============================================================================
# PRUNING (OPTUNA)
# =============================================================================

class PRUNING:
    """Optuna pruning policy (configuration only, no instantiation).
    """

    class OPTUNA:
        # Align with early stopping to avoid pruning on noise
        MIN_EPOCHS: Final[int] = 10

        class MEDIAN:
            N_STARTUP_TRIALS: Final[int] = 10 # same as EARLY_STOP_MIN_EPOCHS
            N_WARMUP_STEPS: Final[int] = 5
            INTERVAL_STEPS: Final[int] = 1


# =============================================================================
# OPTUNA
# =============================================================================

class OPTUNA:
    """Optuna study output and visualization defaults.
    """
    TOP_N_TRIALS: Final[int] = 3
    PLOTS_OUTPUT_DIRNAME: Final[str] = "OPTUNA_graphs"
    PLOTS_NAME_PREFIX: Final[str] = ""


# =============================================================================
# STUDY IDS
# =============================================================================

class STUDY_IDS:
    """Compact human-readable IDs for studies."""
    DIRECT_OPTUNA_PREFIX: Final[str] = "OPT"
    EXHAUSTIVE_SHEET_PREFIX: Final[str] = "XL"
    EXHAUSTIVE_SHEET_RUN_NAME_PREFIX: Final[str] = "excel_study"
    CROSS_VALIDATION_PREFIX: Final[str] = "CV"
    MIN_NUMBER: Final[int] = 100
    MAX_NUMBER: Final[int] = 999
    PLOT_FUNCTIONS: Final[Sequence[str]] = [
        # "parallel_coordinate",
        "slice_plot",
        # "contour_plot",
        "param_importances",
        "intermediate_values",
    ]
# =============================================================================
# CONFIG VALIDATION POLICY
# =============================================================================

class CONFIG:
    """Allowed values used by runtime/config validation schemas."""

    SHARED_FLAT_KEY_PREFIXES: Final[tuple[str, ...]] = (
        "spectrogram.",
        "waveform.",
        "features.",
        "loss.",
        "optimizer.",
        "scheduler.",
    )
    SUPPORTED_RECORDING_CATEGORIES: Final[set[str]] = {"a", "i", "o", "poem"}
    SUPPORTED_TRIM_MODALITIES: Final[set[str]] = {"start", "middle", "end"}
    SUPPORTED_SPEC_NORM: Final[set[str]] = {"minmax_0_1", "minmax_-1_1", "audioset_mean_std"}
    SUPPORTED_FEATURE_NORM: Final[set[str]] = {"minmax_0_1", "minmax_-1_1"}


class SPECTROGRAM_NORM:
    """Central normalization constants for spectrogram/fbank frontends."""

    # AudioSet fbank normalization constants.
    AUDIOSET_FBANK_MEAN: Final[float] = 15.41663
    AUDIOSET_FBANK_STD: Final[float] = 6.55582



# =============================================================================
# LOGGING / FILE CONVENTIONS
# =============================================================================

class LOGGING:
    """Standard logging format policy used across modules."""

    FORMAT: Final[str] = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


class FILES:
    """Standard filenames used across outputs.
    """

    LOGGER_FILENAME: Final[str] = "logger.log"
    # METRICS_JSON_FILENAME: Final[str] = "METRICS" #
