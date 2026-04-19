import sys
import argparse

from src.create_exhaustive_study_sheet import (
    main as run_exhaustive_study_sheet,
)
from src.create_excel_from_pickled_sheet_results import (
    main as run_pickle_to_excel,
)
from src.run_optuna_tuning import optimize
from src.schemas.enums import RunMode
from src.train_autoencoder import run_training as run_autoencoder_training
from src.train_spectrogram import run_training
from src.utils.toolkit.cli import fill_args
from src.run_evaluation_protocol import run_evaluation_protocol
import src.utils.toolkit.time_utils as time_utils


def main() -> int:
    parser = argparse.ArgumentParser(prog="src")
    cmd_args = fill_args(parser)

    if cmd_args.test_run:
        cmd_args.output_dir = f'{cmd_args.output_dir}_{time_utils.get_day_month_year_hour_minute()}'

    if cmd_args.mode == "train":
        if cmd_args.model_type == "autoencoder":
            run_autoencoder_training(cmd_args)
            return 0
        else:
            run_training(cmd_args, run_mode=RunMode.NORMAL_TRAIN_ALWAYS_W_RESULTS)
            return 0

    if cmd_args.mode == "optuna_tuning":
        # setup logger

        _ = optimize(cmd_args, last_optuna_trial_after_tuning=True)
        return 0

    if cmd_args.mode == "exhaustive_study":

        run_exhaustive_study_sheet(cmd_args)

        return 0

    if cmd_args.mode == "cross_val":
        _ = run_evaluation_protocol(cmd_args)
        return 0

    if cmd_args.mode == "pickle_to_excel":
        _ = run_pickle_to_excel(cmd_args)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

