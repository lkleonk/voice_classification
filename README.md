# voice_classification

Spectrogram-based respiratory audio classification for short (~10s) voice recordings. The pipeline turns prepared `.wav` recordings and metadata into trainable audio experiments: spectrogram extraction, scratch CNN/LSTM/ViT models, pretrained audio backbones, demographic baselines, Optuna tuning, grouped evaluation, reporting, and Excel study summaries.

The model stack includes both custom architectures and modern pretrained audio backbones: PANN, PaSST, AST, EfficientAT, wav2vec2, BEATs, and VGGish. This makes the repository useful for comparing lightweight baselines against transfer-learning approaches from AudioSet and self-supervised speech/audio pretraining.

Training can be run on audio/spectrogram inputs alone or with additional demographic and clinical metadata features via `--add_demo_data`; the repository also includes a `tabular_mlp` baseline for metadata-only experiments.

The repository is designed around a clear input boundary: bring recordings plus a metadata CSV in the expected schema, then use the provided training and evaluation machinery to run reproducible model comparisons.

Default workflow:

```
Prepared audio/metadata inputs  -->  Spectrogram extraction / training / evaluation
                               -->  Pickle results per experiment
                               -->  pickle_to_excel  -->  Final Excel summary
```

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

Third-party code under `src/model/efficientat/EfficientAT/` remains under its original MIT license; see `src/model/efficientat/EfficientAT/LICENSE`.

## What This Repository Provides

- End-to-end experiment execution once audio and metadata are prepared.
- Mel-spectrogram model training for CNN, CNN-LSTM, LSTM, and ViT architectures.
- Pretrained audio model wrappers for PANN, PaSST, AST, EfficientAT, wav2vec2, BEATs, and VGGish.
- Optional demographic-feature training through `--add_demo_data` and a demographic-only `tabular_mlp` baseline.
- Patient-aware train/validation/test and evaluation splitting through `audio_id`.
- Optuna hyperparameter search, repeated-seed evaluation, pickled experiment results, and Excel aggregation.
- Local dummy data for fast technical smoke tests.

## Bring Your Own Dataset

This codebase starts from prepared inputs. To use a new dataset, convert your recordings and metadata into the schema consumed by `src/utils/data_preprocessing/demographic_metadata_loader.py`.

Required core columns:

| Column | Meaning |
|--------|---------|
| `audio_sample_path` | Path to the `.wav` file used for training/evaluation |
| `label` | Numeric class label: `0 = control`, `1 = copd` |
| `audio_id` | Patient/group identifier used for patient-aware splitting |
| `age` | Age in years |
| `sex` | `male`, `female`, `m`, or `w` |
| `language` | Language code; currently mapped values are `ru`, `de`, `en`, `es`, `it`, `pt` |

Required binary health/symptom columns, each encoded as `0` or `1`:

```text
angina, asthma, cancer, cystic, diabetes, hbp, heart, hiv, long,
longterm, lung, organ, otherheart, pulmonary, stroke, valvular,
chills, dizziness, drycough, fever, headache, muscleache, runny,
runnyblockednose, shortbreath, smelltasteloss, sorethroat, tightness,
wetcough
```

For a compact working example, see `src/test/test_run_dummy_files/test_metadata_uk_v2.csv`. The bundled test metadata uses the same schema and is what `--test_run` loads by default.

Additional columns are allowed and ignored by the loader. The current public CLI uses the UK-compatible option names (`--dataset_v=uk`, `--recording_category=poem`, `--selected_classes=copd,control`), but your data can still be wired in by matching this metadata contract and passing your CSV through `--metadata_file`.

## Directory Layout

```
src/
|
|-- __main__.py                       # Package dispatcher (CLI entrypoint)
|-- settings.py                       # Hardcoded constants, policy defaults, schema version
|-- train_spectrogram.py              # Core training loop entry (run_training)
|-- train_autoencoder.py              # Autoencoder training (routed through __main__)
|-- run_optuna_tuning.py              # Optuna hyperparameter optimization
|-- run_evaluation_protocol.py        # Evaluation protocol runner
|-- create_exhaustive_study_sheet.py  # Main publication script (tuning + multi-seed eval)
|-- create_excel_from_pickled_sheet_results.py  # Aggregates pickled results into Excel
|
|-- schemas/
|   |-- dataclasses.py                # Core dataclasses (TrainSetup, EvalResult, ...)
|   |-- enums.py                      # RunMode, NormMode, etc.
|   |-- typed_dicts.py                # TypedDict definitions for structured dicts
|
|-- model/
|   |-- model_factory.py              # build_model() registry
|   |-- model_utils.py                # Shared model utilities
|   |-- tabular_mlp.py                # Demographic-only MLP baseline (no audio input)
|   |-- README.md                     # Pretrained checkpoint policy, adding new models
|   |-- efficientat/EfficientAT/      # Vendored EfficientAT source (third-party)
|   |
|   |-- pretrained/
|   |   |-- pretrained_model_embedder_with_clf.py  # Frozen-backbone + classifier head wrapper
|   |   |-- pretrained_models_params.py  # Per-model enforced config defaults
|   |   |-- cache_env.py              # configure_pretrained_cache_env() for TORCH_HOME, HF_HOME, etc.
|   |   |-- pann.py                   # PANN (Cnn14, AudioSet)
|   |   |-- passt.py                  # PaSST (ViT, AudioSet, knowledge-distilled)
|   |   |-- ast.py                    # AST (transformer, AudioSet)
|   |   |-- efficientat_factory.py    # EfficientAT (Dynamic MobileNet, AudioSet)
|   |   |-- wav2vec2.py               # wav2vec2 (self-supervised, raw audio)
|   |   |-- beats.py                  # BEATs (self-supervised)
|   |   |-- vggish.py                 # VGGish (CNN, AudioSet, 1s windows)
|   |
|   |-- scratch/
|   |   |-- scratch_cnn_models.py     # CNN classifier (trained from scratch)
|   |   |-- scratch_cnnlstm.py        # CNN-LSTM hybrid
|   |   |-- scratch_lstm.py           # LSTM sequence classifier
|   |   |-- scratch_vit.py            # Vision Transformer
|   |
|   |-- self_supervised_autoencoder/
|       |-- autoencoder.py            # Autoencoder for self-supervised pretraining
|
|-- utils/
|   |-- data_preprocessing/
|   |   |-- dataset_split_manager.py  # Train/val/test split orchestration
|   |   |-- demographic_metadata_loader.py  # Load + merge demographic columns
|   |   |-- patient_aware_split.py    # Keep all recordings per patient in same fold
|   |   |-- spectrogram_file_dataset.py  # PyTorch Dataset for spectrogram .npy files
|   |
|   |-- reporting/
|   |   |-- demo_data_table.py        # Demographic summary tables
|   |   |-- excel_study.py            # Excel output for exhaustive study results
|   |   |-- logger_setup.py           # Logging configuration
|   |   |-- mlflow_helper.py          # MLflow setup/logging helpers, safe_mlflow_call
|   |   |-- pdf_report.py             # PDF report generation
|   |   |-- training_reporting.py     # Per-run training metrics reporting
|   |
|   |-- toolkit/
|   |   |-- auroc.py                  # Partial AUROC computation
|   |   |-- cli.py                    # Argparse CLI definition
|   |   |-- config_utils.py           # Flat config loading, update_flat_dict_strict, filter/validate
|   |   |-- cuda_handling.py          # GPU device selection and memory management
|   |   |-- eval_model.py             # Model evaluation loop (inference + metrics)
|   |   |-- naming.py                 # Output path and experiment naming conventions
|   |   |-- tempo.py                  # Tempo/timing utilities
|   |   |-- time_utils.py            # Timestamp formatting helpers
|   |
|   |-- training/
|   |   |-- loop.py                   # Main train/val epoch loop
|   |   |-- model_manager.py          # Model checkpoint saving/loading
|   |   |-- setup.py                  # Training setup (optimizer, scheduler, loss)
|   |
|   |-- tuning/
|   |   |-- optuna_reporting.py       # Optuna trial result logging
|   |
|   |-- visualization/
|       |-- gradcam.py                # GradCAM attention maps
|       |-- graph_visualization.py    # Training curve and metric plots
|
|-- test/                             # Pytest tests (unit, integration, slow markers)
|-- pytest.ini                        # Pytest marker definitions
```

Config files:

```
src/config/lk_hyperparameters_spectrogram.yml  # Model/spectrogram/scheduler hyperparameters
src/config/lk_hyperparameters_optuna_tuning.yml  # Optuna search space definitions
```

## Data Requirements

- **Audio**: Short voice recordings (~10s) in `.wav` format.
- **Metadata**: CSV file matching the schema above, with recording paths, labels, patient/group IDs, and demographic/clinical feature columns.
- **Pretrained weights**: Must live under `work/model_weights/<model>/` (project-local, not `~/.cache`). See the checkpoint setup notes below.
- **Dataset adapter**: New datasets should be preprocessed into the expected audio/metadata layout before calling the training CLI.

## Quickstart

### CLI Entrypoint

All modes go through the package dispatcher:

```bash
python -m src --mode=<mode> [args...]
```

Modes: `train`, `optuna_tuning`, `cross_val`, `exhaustive_study`, `pickle_to_excel`

### Local Test Run (Windows)

Lightweight wiring checks with the bundled dummy data, no Snakemake needed:

```powershell
cd path\to\voice_classification
uv run python -m src --mode=train --test_run --model_type=test_cnn
```

Autoencoder:

```powershell
uv run python -m src --mode=train --test_run --model_type=autoencoder
```

### Train With Prepared Data

Pass your schema-compatible metadata file through `--metadata_file`:

```powershell
uv run python -m src --mode=train --metadata_file=data/metadata.csv --model_type=cnn --epochs=20
```

The current public CLI keeps the UK-compatible labels for `--dataset_v`, `--recording_category`, and `--selected_classes`; these are experiment identifiers and validation choices around the metadata contract.

### Publication-Scale Runs

For larger experiments, call the same CLI modes directly from your own scheduler or shell scripts.

Examples:

```bash
uv run --python 3.11 python -m src --mode=optuna_tuning --model_type=cnn
uv run --python 3.11 python -m src --mode=exhaustive_study --test_run
uv run --python 3.11 python -m src --mode=pickle_to_excel
```

Always run with `--test_run` first to validate orchestration before a full launch.

## Models

### Trained from Scratch

| Model | Input | Notes |
|-------|-------|-------|
| `cnn` | Mel-spectrogram | Basic CNN classifier |
| `lstm` | Mel-spectrogram | LSTM sequence classifier |
| `cnnlstm` | Mel-spectrogram | CNN-LSTM hybrid |
| `vit` | Mel-spectrogram | Vision Transformer |

### Pretrained (Fine-tuned)

| Model | Architecture | Pretraining | Input Duration |
|-------|-------------|-------------|----------------|
| `pann` | CNN (Cnn14) | AudioSet | ~10s |
| `passt` | ViT on spectrogram patches | AudioSet (knowledge-distilled) | ~10s |
| `ast` | Transformer | AudioSet | ~10s |
| `efficientat` | Dynamic MobileNet | AudioSet | ~10s |
| `wav2vec2` | Self-supervised | Raw audio (facebook/wav2vec2-large-robust) | ~10s |
| `beats` | Self-supervised | Audio | ~10s |
| `vggish` | CNN | AudioSet | ~1s (exception) |

### Special

| Model | Notes |
|-------|-------|
| `autoencoder` | Self-supervised pretraining, routed through `__main__.py` |
| `tabular_mlp` | Demographic-only baseline, no audio input, requires `--add_demo_data` |

All pretrained checkpoints must be stored under `work/model_weights/<model>/`.

### Checkpoint Setup

Auto-downloaded into project-local cache/work paths:
- `ast`
- `wav2vec2`
- `efficientat`

Manual local checkpoint required:
- `pann`: `work/model_weights/pann/Cnn14_16k_mAP=0.438.pth`
- `passt`: `work/model_weights/passt/passt-s-kd-ap.486.pt`
- `vggish`: `work/model_weights/vggish/vggish-10086976.pth`
- `beats`: `work/model_weights/beats/BEATs_iter3.pt`

If a manual-checkpoint model is missing its expected local file, the code raises `FileNotFoundError` with the required path.

Pinned source identifiers used by autodownloaded models:
- `ast`: `MIT/ast-finetuned-audioset-10-10-0.4593`
- `wav2vec2`: `facebook/wav2vec2-large-robust`
- `efficientat`: `dymn20_as_mAP_493.pt`

## Config Contract

The training/tuning config is a **flat dictionary** with dot-notated string keys:

```
spectrogram.n_mels
scheduler.step_size
model.dropout
```

This flat structure is intentional and must not be converted to nested dicts. It flows through spectrogram, scheduler, and model/training hyperparameters.

Config flow by mode:
- **Direct training**: `run_training(args, config=None)` loads from `args.config_file`.
- **Optuna**: Creates `updated_trial_config = update_flat_dict_strict(base_config, param_space)`, passes to `run_training(args, config=updated_trial_config, trial=trial)`.
- **Cross-validation**: Loads config once, passes `config=config` into each fold.
- **Exhaustive study**: Sets `new_args.config_file` to best-trial YAML; downstream reads that path.

`settings.py` holds hardcoded constants and policy defaults. Values there are stable and must not change at runtime.

Prefer strict key access (`config["key"]`) over `.get("key", default)` -- this is research code, fail fast on missing values.

## Schema Version

`CURRENT_SCHEMA_VERSION` in `settings.py` defines the version tag for persisted experiment artifacts (best-config YAMLs, pickled results). Change this tag when the output structure or interpretation changes and new outputs should be kept separate from older artifacts.

## Testing

Run from the `voice_classification` root so `src...` imports resolve:

```bash
# Single test file
uv run --python 3.11 python -m pytest -p no:cacheprovider src/test/test_toolkit_auroc.py -q

# Multiple files
uv run --python 3.11 python -m pytest -p no:cacheprovider src/test/test_toolkit_naming.py src/test/test_toolkit_config_utils.py -q

# Verbose debugging
uv run --python 3.11 python -m pytest -p no:cacheprovider src/test/test_toolkit_auroc.py -vv -ra -s

# Markers only
uv run --python 3.11 python -m pytest -p no:cacheprovider src/test -m unit -q
```

Markers (defined in `pytest.ini`): `unit`, `integration`, `slow`.

Note: `-p` is for pytest plugins (e.g., `-p no:cacheprovider`). Do not pass a test file path to `-p`.

## Linting

Uses `ruff`. Run scoped to the package:

```bash
cd src/
ruff check .
```

## Reference Dataset And Scope

The default public metadata schema follows the UK-compatible COPD/control setup. The related public dataset is the UK-COVID voice dataset:

https://zenodo.org/records/11167750

The repository is intentionally distributed at the prepared-input stage: it contains the training, tuning, evaluation, reporting, and aggregation pipeline, while dataset-specific upstream preprocessing should be wired by the user to produce the metadata schema documented above. The bundled dummy data makes fast technical validation possible without downloading a full dataset.

## Implementation Notes

- **Reproducibility is not fully deterministic.** `args.random_seed` controls CV splits but: Python/NumPy/Torch RNGs are not globally seeded, DataLoader uses `shuffle=True` without a generator/worker_init_fn, train augmentations (FrequencyMasking, TimeMasking, Gaussian noise) use unseeded randomness, and CUDA deterministic flags are not set.
- **Reporting behavior differs by mode.** Full per-run report saving activates with `RunMode.NORMAL_TRAIN_ALWAYS_W_RESULTS` (direct `train_spectrogram.py` calls). Optuna and CV focus on tuning/evaluation artifacts.
