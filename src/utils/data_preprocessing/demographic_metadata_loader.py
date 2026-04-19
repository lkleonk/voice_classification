import json
import os

import librosa

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

import src.utils.toolkit.time_utils as time_utils


UK_BINARY_FEATURE_COLUMNS = [
    "angina", "asthma", "cancer", "cystic", "diabetes", "hbp", "heart",
    "hiv", "long", "longterm", "lung", "organ", "otherheart", "pulmonary",
    "stroke", "valvular", "chills", "dizziness", "drycough", "fever",
    "headache", "muscleache", "runny", "runnyblockednose", "shortbreath",
    "smelltasteloss", "sorethroat", "tightness", "wetcough",
]
UK_REQUIRED_METADATA_COLUMNS = (
    "audio_sample_path",
    "label",
    "audio_id",
    "age",
    "sex",
    "language",
    *UK_BINARY_FEATURE_COLUMNS,
)
UK_LABEL_MAPPING = {0: "control", 1: "copd"}

#from __future__ import annotations
def get_norm_off_filepath(filepath):
    try:
        # Get the parent directory of the file
        parent_dir = os.path.dirname(filepath)
        # Get the grandparent directory
        grandparent_dir = os.path.dirname(parent_dir)
        # Get the last part of the parent directory
        parent_folder_name = os.path.basename(parent_dir)

        # Replace '_on' at the end with '_off'
        if parent_folder_name.endswith('_on'):
            new_folder_name = parent_folder_name[:-3] + '_off'
        else:
            raise ValueError(f"Parent folder name does not end with '_on': {parent_folder_name}")

        # Construct the new path
        new_path = os.path.join(grandparent_dir, new_folder_name, os.path.basename(filepath))
        return new_path

    except Exception as e:
        print(f"Error: {e}")
        return None


def validate_uk_metadata_schema(metadata: pd.DataFrame, selected_conditions_list: list[str]) -> None:
    missing_columns = [
        column_name for column_name in UK_REQUIRED_METADATA_COLUMNS if column_name not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required UK metadata columns: "
            + ", ".join(sorted(missing_columns))
        )

    supported_conditions = set(UK_LABEL_MAPPING.values())
    unknown_conditions = sorted(set(selected_conditions_list) - supported_conditions)
    if unknown_conditions:
        raise ValueError(
            "UK-only metadata loader supports only these conditions: "
            f"{sorted(supported_conditions)}. "
            f"Received: {unknown_conditions}"
        )


"""

on loading the newer coviduk19 data w copd:

# available:  audio_sample_path, age, sex, height_cm - maybe include language: - collect automatically and then one-hot-encoding category

demographic_data_dict = {
    row["audio_sample_path"]: {
        "age": row.get("age", None),
        "sex": row.get("sex", None),
        "language": row.get("language", None),
    }
    for _, row in filtered_metadata.iterrows()
}

"""


# --- Missingness indicators ---
def get_missigness_indicator(column_data) -> np.ndarray:
    """
    Returns a binary indicator (float32) showing whether each value is present (1) or missing (0).
    """
    return np.asarray(~pd.isna(column_data), dtype=np.float32) # (~np.isnan(bmis)).astype(np.float32)

def load_data_from_metadata_csv(metadata_file, recording_category, selected_conditions, features_norm_mode, logger = None, demo_data_mode = None, add_acoustic_features = False): # load_data.py
    # Load metadata
    metadata = pd.read_csv(metadata_file, index_col=False)
    if logger: 
        #logger.info(f'titles of the loaded metadata.csv before any processing: {metadata.head(1)}')
        logger.info(f"Raw columns: {metadata.columns.tolist()}")
        logger.info(f"First row: {metadata.iloc[0].tolist()}")
        logger.info(f"Number of columns: {len(metadata.columns)}")
        logger.info(f"First row length: {len(metadata.iloc[0])}")
    # Clean column names by removing trailing semicolons
    metadata.columns = [col.strip().rstrip(';') for col in metadata.columns]
    selected_conditions_list = [cls.strip().lower() for cls in selected_conditions.split(',') if cls.strip()]
    validate_uk_metadata_schema(metadata, selected_conditions_list)

    filtered_metadata = metadata.copy()
    filtered_metadata["label_name"] = filtered_metadata["label"].map(UK_LABEL_MAPPING)
    if filtered_metadata["label_name"].isna().any():
        unexpected_labels = sorted(filtered_metadata.loc[filtered_metadata["label_name"].isna(), "label"].unique().tolist())
        raise ValueError(
            "UK metadata contains unsupported label values. "
            f"Expected only {sorted(UK_LABEL_MAPPING)} but found {unexpected_labels}"
        )

    available_conditions = set(filtered_metadata["label_name"].unique())
    missing_conditions = [c for c in selected_conditions_list if c not in available_conditions]
    if missing_conditions:
        raise ValueError(f"The following selected conditions are missing from the UK metadata: {missing_conditions}")

    filtered_metadata = filtered_metadata[
        filtered_metadata["label_name"].isin(selected_conditions_list)
    ].copy()
    if filtered_metadata.empty:
        raise ValueError("No rows left after filtering UK metadata for the selected conditions.")

    further_data_column_names = list(UK_BINARY_FEATURE_COLUMNS)
    base_fields = ["age", "sex", "language"]
    all_fields = base_fields + further_data_column_names

    demographic_data_dict = {
        row["audio_sample_path"]: {field: row[field] for field in all_fields}
        for _, row in filtered_metadata.iterrows()
    }

    label_encoder = LabelEncoder()
    filtered_metadata["class_idx"] = label_encoder.fit_transform(filtered_metadata["label_name"])
    group_ids = np.asarray(filtered_metadata["audio_id"])

    if logger:
        logger.info("Validated UK-only metadata schema with %d rows.", len(filtered_metadata))

    # Extract filepaths and labels
    filepaths = [str(p) for p in filtered_metadata["audio_sample_path"].tolist()]

    if not filepaths:
        if logger:
            logger.error("No filepaths found in the filtered metadata.")
        raise ValueError("No filepaths found in the filtered metadata. Check your selections and metadata.csv.")

    ending = filepaths[0][-4:]
    if ending != '.wav':
        raise ValueError(f'File path 1 does not end on .wav (instead: {ending}) - check metadata.csv')


    # --- DEBUG: log file endings ---
    N = min(5, len(filepaths))
    if logger:
        logger.info(
            "Sample file endings: %s",
            [(fp[-10:], fp[-4:]) for fp in filepaths[:N]]
        )
    
    labels = filtered_metadata["class_idx"].values


    # === Demographics postprocessing - normalizing data ===
    # Example mapping for sex
    SEX_MAPPING = {"male": 0, "female": 1, "m": 0, "w": 1}
    MAX_AGE_YEARS = 100.0

    # Extract ages and sexes
    
    sexes_strings = np.asarray(
        filtered_metadata["sex"].astype(str).str.lower().replace("nan", np.nan),
        dtype=object,
    )  # needed to create sexes_mask
    sexes_norm = np.asarray(
        filtered_metadata["sex"].str.lower().map(SEX_MAPPING).fillna(0.5),
        dtype=np.float32,
    )
    ages = np.asarray(filtered_metadata["age"], dtype=float)

    #sexes_mask = (~pd.isna(sexes_strings)).astype(np.float32)  # 1 if present, 0 if missing
    #ages_mask  = (~pd.isna(ages)).astype(np.float32)

    sexes_mask = get_missigness_indicator(sexes_strings)
    ages_mask  = get_missigness_indicator(ages)

    check_for_missing_demo_data(sexes_mask, "sexes", logger)
    check_for_missing_demo_data(ages_mask,  "ages",  logger)

    # Normalize ages: 0-100 years maps to 0-1, anything over 100 clamped to 1
    ages_norm = np.clip(np.nan_to_num(ages, nan=0.5 * MAX_AGE_YEARS) / MAX_AGE_YEARS, 0.0, 1.0)

    LANGUAGE_MAPPING = {"ru": 0.0, "de": 0.2, "en": 0.4, "es": 0.6, "it": 0.8, "pt": 1.0}
    language_strings = np.asarray(
        filtered_metadata["language"].astype(str).str.lower().replace("nan", np.nan),
        dtype=object,
    )
    languages_norm = np.asarray(
        filtered_metadata["language"].astype(str).str.lower().map(LANGUAGE_MAPPING).fillna(0.5),
        dtype=np.float32,
    )
    languages_mask = get_missigness_indicator(language_strings)

    check_for_missing_demo_data(languages_mask, "languages", logger)

    def get_uk_data_masks_and_values(data, column_names) -> np.ndarray:
        stacked_data: list[np.ndarray] = []
        for col_name in column_names:
            col_value = data[col_name]

            if not col_value.isin([0, 1]).all():
                raise ValueError(f"UK metadata column '{col_name}' contains values other than 0 or 1")

            col_value_np = np.asarray(col_value, dtype=np.float32)
            missingness_indicator = get_missigness_indicator(col_value_np)
            stacked_data.append(missingness_indicator)
            stacked_data.append(col_value_np)

        return np.column_stack(stacked_data)

    further_data_masks_and_values = get_uk_data_masks_and_values(filtered_metadata, further_data_column_names)

    demographic_features = np.column_stack((
        ages_mask, ages_norm,
        sexes_mask, sexes_norm,
        languages_mask, languages_norm,
        further_data_masks_and_values
    )).astype(np.float32)

    # Convert to PyTorch tensor
    features_tensor = torch.tensor(demographic_features, dtype=torch.float32)
    print(features_tensor.shape)



    if features_norm_mode == 'minmax_-1_1':
        # Convert from 0-1 to -1 to 1
        features_tensor = features_tensor * 2 - 1
    elif features_norm_mode == 'minmax_0_1':
        # Already in 0-1 range, do nothing
        pass
    else:
        raise ValueError(f"features.norm '{features_norm_mode}' unknown")


    return filepaths, labels, label_encoder, demographic_data_dict, features_tensor, group_ids





def sanity_check_data(paths, labels, l_encoder, demo_dict, demo_tensor, logger = None, idx = 0):  # load_data.py
    decoded_label = l_encoder.inverse_transform([labels[idx]])[0]

    info_msg = (
        f"\nSample index: {idx}\n"
        f"File path: {paths[idx]}\n"
        f"Label: {labels[idx]}\n"
        f"Decoded label: {decoded_label}\n"
        f"Demographics (dict): {demo_dict[paths[idx]]}\n"
        f"Demographics (tensor): {demo_tensor[idx].tolist()}"
    )
    if logger:
        logger.info(info_msg)
    else:
        print(info_msg)





def check_for_missing_demo_data(mask, name, logger=None):
    """
    Logs or prints the number of missing entries in a mask array.

    Parameters:
    -----------
    mask : array-like
        A boolean or numeric mask where 1/True indicates presence and 0/False indicates missing.
    name : str
        The name of the feature or column being checked (used in the output message).
    logger : logging.Logger, optional
        If provided, uses logger.info to log the message. Otherwise, prints to stdout.

    Example:
    --------
    check_for_missing_data(sexes_mask, "sexes", logger)
    """
    missing = int((mask == 0).sum())
    total   = len(mask)
    msg = f"Missing {name}: {missing}/{total}"
    if logger:
        logger.info(msg)
    else:
        print(msg)




    
def convert_to_serializable(data):                
    """
    Recursively converts any PyTorch Tensors in the data structure to lists
    to make them JSON serializable.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().tolist()  # Detach, move to CPU, and convert Tensor to list
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]  # Recursively handle lists
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}  # Recursively handle dicts
    else:
        return data  # Return as is for other types






def save_loss_as_json(output_dir, file_name, train_loss_list, val_loss_list):     
    # Ensure that all tensors in train_loss_list and val_loss_list are converted
    losses_dictionary = {
        "train_loss": convert_to_serializable(train_loss_list),
        "val_loss": convert_to_serializable(val_loss_list),
    }

    # Save in model_train_run_info - including model type and time stamp:
    time_stamp = time_utils.get_day_month_year_hour_minute()  # This is assumed to be a utility function you have
    json_file_name = os.path.join(output_dir, f"{file_name}_{time_stamp}.json")

    with open(json_file_name, 'w') as json_file:
        json.dump(losses_dictionary, json_file)

    print(f"Losses saved as JSON at: {json_file_name}")

    return json_file_name







def get_stats_on_sample_duration(X_paths):         
    """
    Calculate statistics on the duration of audio samples using librosa.
    
    Args:
        X_paths (list): List of paths to .wav files
        
    Returns:
        tuple: (average_duration, median_duration, standard_deviation) in seconds
    """
    durations = []
    #print(X_paths)
    for file_path in X_paths:
        # Get duration using librosa
        duration = librosa.get_duration(path=file_path)
        durations.append(duration)
            
    avg_dur = float(np.mean(durations))
    median_dur = float(np.median(durations))
    std_dur = float(np.std(durations))
    
    return avg_dur, median_dur, std_dur











