from __future__ import annotations

from pathlib import Path


def require_local_checkpoint(
    checkpoint_path: Path,
    *,
    model_name: str,
    source_hint: str | None = None,
) -> Path:
    """Return a required local checkpoint path or raise a clear setup error."""
    if checkpoint_path.is_file():
        return checkpoint_path

    message = (
        f"Missing local checkpoint for '{model_name}'. "
        f"Expected file at: {checkpoint_path}. "
        "This model is not auto-downloaded by the repository. "
        "Download the checkpoint manually and place it at the expected path."
    )
    if source_hint:
        message += f" Source: {source_hint}."
    raise FileNotFoundError(message)
