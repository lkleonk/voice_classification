from __future__ import annotations
import os
from src.settings import PATHS


def configure_pretrained_cache_env() -> dict[str, str]:
    """
    Keep all pretrained-model caches inside project/work paths.

    Uses canonical locations from settings.PATHS.
    Uses setdefault to respect externally provided environment variables.
    """
    resolved_base_dir = PATHS.WORK_DIR.resolve()
    resolved_model_weights_dir = PATHS.MODEL_WEIGHTS_DIR.resolve()

    cache_paths = {
        "TORCH_HOME": str(resolved_model_weights_dir),
        "HF_HOME": str(resolved_base_dir / "_hf_home"),
        "TRANSFORMERS_CACHE": str(resolved_base_dir / "_transformers_cache"),
        "HF_HUB_CACHE": str(resolved_base_dir / "_hf_hub_cache"),
        "HUGGINGFACE_HUB_CACHE": str(resolved_base_dir / "_hf_hub_cache"),
        "XDG_CACHE_HOME": str(resolved_base_dir / "_xdg_cache_home"),
    }

    for key, value in cache_paths.items():
        os.environ.setdefault(key, value)

    return cache_paths
