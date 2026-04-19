import argparse

from src.settings import CLI_CHOICES


def fill_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--test_run", action="store_true", help="Implement test run or not")

    # Data and directories
    parser.add_argument("--mode", choices=CLI_CHOICES.MODES, help="Execution mode")
    parser.add_argument("--output_dir", type=str, default="outputs/test_output_dir_norm_off", help="Directory to save outputs (results, plots).")
    parser.add_argument("--metadata_file", type=str, default="src/test/test_run_dummy_files/test_metadata_uk_v2.csv", help="Path to the metadata CSV file.")
    parser.add_argument("--tuning_file", type=str, default="src/config/lk_hyperparameters_optuna_tuning.yml", help="Path to the optuna parameter tuning file.")
    parser.add_argument("--config_file", type=str, default="src/config/lk_hyperparameters_spectrogram.yml", help="Path to the configuration for the spectrogram creation and architecture parameters.")
    parser.add_argument("--selected_classes", type=str, default="copd,control", choices=CLI_CHOICES.SELECTED_CLASSES, help="Comma-separated string of classes to include.")
    parser.add_argument("--recording_category", type=str, default="poem", choices=CLI_CHOICES.RECORDING_CATEGORY, help="Recording category. UK-only release supports 'poem'.")
    parser.add_argument("--add_demo_data", action="store_true", help="Flag for whether to add demographic data in the training pipeline")


    # Model and training parameters
    parser.add_argument("--model_type", type=str, default="test_cnn", choices=CLI_CHOICES.MODEL_TYPES, help="Model type trained.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs.")
    parser.add_argument("--trials", type=int, default=4, help="Number of trials - only relevant for optuna.")
    parser.add_argument("--dataset_v", type=str, default="uk", choices=CLI_CHOICES.DATASET_V, help="Dataset version. This open-source release supports only 'uk'.")
    parser.add_argument("--norm_mode", type=str, default="norm_off", choices=CLI_CHOICES.NORM_MODE, help="Normalization of audio mode: 'norm_on' or 'norm_off'.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    cli_args = parser.parse_args()

    # Override parameters if test_run is True
    if cli_args.test_run:  # relevant for when working on the server
        cli_args.trials = 2
        cli_args.epochs = 2
        cli_args.metadata_file = "src/test/test_run_dummy_files/test_metadata_uk_v2.csv"

    return cli_args
