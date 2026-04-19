import copy
import os
from typing import Any, Dict

import yaml

from src.model.pretrained.pretrained_models_params import (
    get_pretrained_model_spec,
)
from src.settings import CONFIG, PATHS


def old_load_config_via_yaml_from_other_run(
    fully_pretrained_model_file_name: str,
    args,
):
    ### specific usecase of using fully pretrained model - same settings (config and some args entries) are needed
    # handle loading the config of the pretrained clf --> reason: same dataset and model settings are needed.
    new_config_name = fully_pretrained_model_file_name.replace('.pt','.yml') # yaml has the same name as .pth file - so just replacing file ending is sufficient
    full_path = os.path.join(PATHS.MODEL_WEIGHTS_DIR, new_config_name)

    with open(full_path, "r", encoding="utf-8") as f:
        model_settings = yaml.safe_load(f)

    new_config = model_settings['configuration']
    args_dict = model_settings['arguments']
    
    args.model_type = args_dict['model_type']
    args.dataset_v = args_dict['dataset_v']
    args.selected_classes = args_dict['selected_classes']
    args.recording_category = args_dict['recording_category']
    args.metadata_file = args_dict['metadata_file']
    args.config_file = args_dict['config_file']
    args.pretrained_registry = args_dict['pretrained_registry']
    args.random_seed = args_dict['random_seed']
    args.trials = args_dict['trials']
    args.test_run = args_dict['test_run']

    # Add any missing args keys (some older configs won't have new fields)
    for key, value in args_dict.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    new_config[f'{args.model_type}.fully_pretrained_model_file_name'] = fully_pretrained_model_file_name # because the loaded config is old and doesnt contain the correct pretrained model file name

    return new_config





def update_flat_dict_strict(base_config: Dict[str, Any], target_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same as update_flat_dict_strict, but raises immediately on the first unknown key encountered.
    """
    updated = copy.deepcopy(base_config)

    for k, v in target_values.items():
        if k not in updated:
            raise KeyError(f"Strict update refused: unknown key '{k}'")
        updated[k] = v

    return updated




def compare_search_space_and_config(config: dict, search_space: dict) -> None:
    """
    Verify that every key in `search_space` exists in the flat `config` dict.

    Raises ValueError listing all missing keys.
    """
    missing = [k for k in search_space if k not in config]

    if missing:
        missing_str = ", ".join(f"'{k}'" for k in sorted(missing))
        raise ValueError(
            "Configuration validation error: the following search-space keys "
            f"do not exist in the flat config: {missing_str}"
        )


def validate_and_clean_tuning_config(
    tuning_config: Dict[str, Any],
    model_type: str,
    dataset_v: str,
    logger: Any,
    strict: bool,
) -> Dict[str, Any]:
    """
    Validate and sanitize Optuna search-space for a specific run.

    Steps:
    - Keep only keys relevant for this run:
      `<model_type>.*` and shared namespaces from CONFIG.SHARED_FLAT_KEY_PREFIXES.
    - Remove keys that will be overwritten by pretrained model spec logic.
    """
    cleaned = dict(tuning_config)
    model_key = (model_type or "").lower()
    dataset_key = (dataset_v or "").lower()

    relevant_prefixes = (f"{model_key}.", *CONFIG.SHARED_FLAT_KEY_PREFIXES)
    irrelevant_keys = [k for k in cleaned if not k.startswith(relevant_prefixes)]
    for key in irrelevant_keys:
        cleaned.pop(key, None)
    if irrelevant_keys:
        logger.info(
            "Dropped %d irrelevant tuning keys (allowed prefixes: %s).",
            len(irrelevant_keys),
            ", ".join(relevant_prefixes),
        )

    try:
        enforced_spec_keys = set(get_pretrained_model_spec(model_key).keys())
    except KeyError:
        enforced_spec_keys = set()

    overwritten_tuned_keys = sorted(k for k in cleaned if k in enforced_spec_keys)
    if overwritten_tuned_keys:
        if strict:
            joined = ", ".join(overwritten_tuned_keys)
            raise ValueError(
                "Tuning config contains keys that are overwritten by pretrained model defaults: "
                f"{joined}"
            )
        for key in overwritten_tuned_keys:
            cleaned.pop(key, None)
            logger.warning(
                "Removed tuning key '%s' because pretrained model '%s' overwrites it.",
                key,
                model_key,
            )

    if model_key == "tabular_mlp":
        spectrogram_keys = [k for k in cleaned if k.startswith("spectrogram.")]
        for key in spectrogram_keys:
            cleaned.pop(key, None)
        if spectrogram_keys:
            logger.info(
                "Removed %d spectrogram tuning keys for model '%s' because it does not consume audio inputs.",
                len(spectrogram_keys),
                model_key,
            )


    logger.info(
        "Validated/cleaned tuning config for model=%s dataset=%s: %d -> %d keys.",
        model_key,
        dataset_key,
        len(tuning_config),
        len(cleaned),
    )
    return cleaned



