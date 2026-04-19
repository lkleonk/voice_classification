from __future__ import annotations

from typing import Any

from src.settings import PATHS, SPECTROGRAM_NORM


# Flat per-model specs used as the single source of truth for model-specific enforced parameters.
_PRETRAINED_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "vggish": {
        "vggish.embed_dim": 128,
        "spectrogram.duration": 0.95,
        "spectrogram.sample_rate": 16000,
        "spectrogram.window_length": 400,
        "spectrogram.n_fft": 400,
        "spectrogram.hop_length": 160,
        "spectrogram.n_mels": 64,
        "spectrogram.fmin": 125,
        "spectrogram.fmax": 7500,
        "spectrogram.norm": "audioset_mean_std", 
    },
    "pann": {
        "pann.embed_dim": 2048,
        "spectrogram.duration": 10,
        "spectrogram.sample_rate": 16000,
        "spectrogram.window_length": 512,
        "spectrogram.n_fft": 512,
        "spectrogram.hop_length": 160,
        "spectrogram.n_mels": 64,
        "spectrogram.fmin": 50,
        "spectrogram.fmax": 8000,
        "spectrogram.norm": "minmax_-1_1",
    },
    "passt": {
        "passt.embed_dim": 768,
        "spectrogram.duration": 10,
        "spectrogram.sample_rate": 32000,
        "spectrogram.window_length": 800,
        "spectrogram.hop_length": 320,
        "spectrogram.n_mels": 128,
        "spectrogram.fmin": 50,
        "spectrogram.fmax": 14000,
        "passt.n_fft": 1024,
        "passt.input_tdim": 998,
    },
    "efficientat": {
        "efficientat.embed_dim": 1920,
        "spectrogram.duration": 10,
        "spectrogram.sample_rate": 32000,
        "spectrogram.window_length": 800,
        "spectrogram.n_fft": 1024,
        "spectrogram.hop_length": 320,
        "spectrogram.n_mels": 128,
        "spectrogram.fmin": 0,
        "spectrogram.fmax": 15000,
    },
    "ast": {
        "ast.embed_dim": 768,
        # AudioSet clips are 10 s; log-mel features are later padded/truncated to 1024 frames for AST
        "spectrogram.duration": 10,           
        "spectrogram.sample_rate": 16000,

        # framing (Kaldi fbank: 25ms/10ms)
        "spectrogram.window_length": 400,        # 25 ms @ 16 kHz
        "spectrogram.hop_length": 160,           # 10 ms @ 16 kHz

        # FFT: en implementaciones Kaldi-compliance suele redondear a potencia de 2
        "spectrogram.n_fft": 512,

        # mel bins and range
        "spectrogram.n_mels": 128,
        "spectrogram.fmin": 20,
        "spectrogram.fmax": 8000,

        # important for AST
        "ast.input_tdim": 1024,                  # pad/trunc a 1024 frames
        "spectrogram.norm": "audioset_mean_std", 
        "spectrogram.mean": -4.2677393,
        "spectrogram.std": 4.5689974,
    },

    "wav2vec2.0": {
        "wav2vec2.embed_dim": 768,
        "wav2vec2.model_id": PATHS.WAV2VEC2_VARIANT_ID,
        "spectrogram.duration": 10,
        "spectrogram.sample_rate": 16000,
        # wav2vec2 uses raw waveform internally,
        # but keeping these for pipeline compatibility:
        "spectrogram.window_length": 400,
        "spectrogram.n_fft": 400,
        "spectrogram.hop_length": 160,
        "spectrogram.n_mels": 128,
        "spectrogram.fmin": 50,
        "spectrogram.fmax": 8000,
    },
    "beats": {
        "beats.embed_dim": 768,

        # audio (important)
        "spectrogram.duration": 10,
        "spectrogram.sample_rate": 16000,

        # Kaldi-style fbank framing (internal in BEATs)
        "spectrogram.window_length": 400,    # 25 ms @ 16k
        "spectrogram.hop_length": 160,       # 10 ms @ 16k
        "spectrogram.n_fft": 512,            # power-of-2 FFT (Kaldi style)

        # mel config
        "spectrogram.n_mels": 128,
        "spectrogram.fmin": 20,
        "spectrogram.fmax": 8000,

        # normalization (AudioSet style, applied to log-mel features)
        "spectrogram.norm": "audioset_mean_std",
        "beats.fbank_mean": SPECTROGRAM_NORM.AUDIOSET_FBANK_MEAN,
        "beats.fbank_std": SPECTROGRAM_NORM.AUDIOSET_FBANK_STD,
    },
}
# Backward-compatible alias used by model_type/registry naming.
_PRETRAINED_MODEL_SPECS["wav2vec2"] = dict(_PRETRAINED_MODEL_SPECS["wav2vec2.0"])

def get_pretrained_model_spec(model_name: str) -> dict[str, Any]:
    model_key = model_name.lower()
    if model_key not in _PRETRAINED_MODEL_SPECS:
        raise KeyError(f"Unknown pretrained model spec: {model_name}")
    return dict(_PRETRAINED_MODEL_SPECS[model_key])


def overwrite_config_values(config: dict, model_name: str) -> dict:
    model_key = model_name.lower()
    if model_key not in _PRETRAINED_MODEL_SPECS:
        return config

    # in any case:
    config["spectrogram.add_chromagram"] = False

    # for both: # float32, normalized between roughly -1.0 and 1.0
    if model_key == "vggish":  # 95 x 64
        # Keep explicit branch comments for readability in this central source.
        pass
    elif model_key == "pann":  # 96 frames × 64 mel bins
        # not that relevant because when using pann, we use the spectrogram creation function of pann itself
        pass
    elif model_key == "passt":
        # These changes are cosmetic as thhis script uses the internal model.mel module to create spectrograms when using the PaSST model
        # PaSST defaults  these produce ~998 frames for a 10s clip in the repo (i.e. ~10s input duration).
        # See PaSST repo README / config (models.mel.* and models.net.*) for details.
        pass
    elif model_key == "efficientat":
        # Cosmetic alignment for logging/config snapshots.
        # EfficientAT uses its own AugmentMelSTFT frontend in efficientat_factory.py.
        pass
    elif model_key == "ast":
        # AST uses raw waveform input and its own torchaudio frontend in model/ast.py.
        # These values align shared config snapshots and waveform trimming behavior.
        pass
    for key, value in _PRETRAINED_MODEL_SPECS[model_key].items():
        config[key] = value

    return config

