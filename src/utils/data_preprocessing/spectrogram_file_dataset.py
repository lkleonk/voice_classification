import random
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from src.settings import SPECTROGRAM_NORM

def simple_minmax_norm(x, method='minmax_0_1', torch_tensor=False):
    """
    Linearly normalize an array or tensor to a specified range.

    Parameters:
        x (np.ndarray or torch.Tensor): Input data.
        method (str): Normalization method, either 'minmax_0_1' or 'minmax_-1_1'.
        torch_tensor (bool): If True, x must be a torch.Tensor; if False, x must be a np.ndarray.

    Returns:
        np.ndarray or torch.Tensor: Normalized data.
    """
    # Check input type matches torch_tensor flag
    if torch_tensor and not isinstance(x, torch.Tensor):
        raise ValueError("torch_tensor=True but input is not a torch.Tensor")
    if not torch_tensor and not isinstance(x, np.ndarray):
        raise ValueError("torch_tensor=False but input is not a np.ndarray")

    # Normalize to [0, 1]
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min + 1e-9)

    # Adjust to [-1, 1] if needed
    if method == 'minmax_0_1':
        return x_norm
    elif method == 'minmax_-1_1':
        return x_norm * 2 - 1
    else:
        raise ValueError("method must be 'minmax_0_1' or 'minmax_-1_1'")


def normalize_spectrogram_values(
    x,
    norm_mode='minmax_0_1',
    *,
    torch_tensor=False,
):
    """
    Normalize spectrogram-like values. Supports min-max modes and AudioSet mean/std normalization.
    """
    if norm_mode in {'minmax_0_1', 'minmax_-1_1'}:
        return simple_minmax_norm(x, norm_mode, torch_tensor=torch_tensor)

    if norm_mode == 'audioset_mean_std':
        return (x - float(SPECTROGRAM_NORM.AUDIOSET_FBANK_MEAN)) / float(SPECTROGRAM_NORM.AUDIOSET_FBANK_STD)

    raise ValueError(
        "norm_mode must be 'minmax_0_1', 'minmax_-1_1', or 'audioset_mean_std'"
    )


def z_norm(x, axis=None, eps=1e-9, to_zero_one=False):
    """
    Normalizes an array either with z-score or min-max scaling.

    Parameters:
        x (np.ndarray): Input array.
        axis (int or tuple of ints, optional): Axis or axes along which to compute normalization.
                                               Default is None (normalize entire array).
        eps (float): Small value to avoid division by zero (for z-score).
        to_zero_one (bool): If True, normalize values to the range [0, 1] instead of z-score.

    Returns:
        np.ndarray: Normalized array.
            - If to_zero_one=False: Z-score normalized, centered around 0 with std 1 (range typically [-3, 3]).
            - If to_zero_one=True: Min-max normalized to range [0, 1].
    """
    if to_zero_one:
        min_val = x.min(axis=axis, keepdims=True)
        max_val = x.max(axis=axis, keepdims=True)
        return (x - min_val) / (max_val - min_val + eps)
    else:
        mean = x.mean(axis=axis, keepdims=True)
        std = x.std(axis=axis, keepdims=True) + eps
        return (x - mean) / std







def add_gaussian_noise(wav, max_noise_std):
    """
    Adds Gaussian noise to a waveform.

    Args:
        wav: waveform, np.ndarray or torch.Tensor
             shape (n_samples,) or (1, n_samples)
        max_noise_std: maximum std of Gaussian noise.
                       If <= 0, no noise is added.

    Returns:
        noisy_wav: same type and shape as input
    """
    # -------------------------------------------------
    # Early exit: noise disabled
    # -------------------------------------------------
    if max_noise_std <= 0:
        return wav

    # Random noise level for this sample
    noise_std = random.uniform(1e-7, max_noise_std)

    if isinstance(wav, np.ndarray):
        noise = np.random.normal(0.0, noise_std, wav.shape)
        return wav + noise

    elif isinstance(wav, torch.Tensor):
        noise = noise_std * torch.randn_like(wav)
        return wav + noise

    else:
        raise TypeError(
            f"Expected np.ndarray or torch.Tensor, got {type(wav)}"
        )



def create_spectrogram_from_waveform_w_config(waveform, config):
    return create_spectrogram_from_waveform(
        waveform=waveform,
        sample_rate=config["spectrogram.sample_rate"],
        n_mels=config["spectrogram.n_mels"],
        n_fft=config["spectrogram.n_fft"],
        window_length=config.get("spectrogram.window_length", config["spectrogram.n_fft"]),
        hop_length=config["spectrogram.hop_length"],
        fmin=config["spectrogram.fmin"],
        fmax=config["spectrogram.fmax"],
        duration=config["spectrogram.duration"],
        trim_modality=config["spectrogram.trim_modality"],
        add_chromagram=config["spectrogram.add_chromagram"], # either True or False
        add_mfcc=config["spectrogram.add_mfcc"], # either True or False
        add_delta_mfcc=config["spectrogram.add_delta_mfcc"], # either True or False
        norm_modality=config["spectrogram.norm"], # either 'minmax_0_1', 'minmax_-1_1', 'audioset_mean_std'

    )


def create_spectrogram_from_waveform(
    waveform,
    sample_rate,
    n_mels,
    n_fft,
    window_length,
    hop_length,
    fmin,
    fmax,
    duration,
    trim_modality,
    add_chromagram,
    add_mfcc,
    add_delta_mfcc,
    norm_modality,
):
    """
    Converts a single audio waveform into a log-mel spectrogram of fixed shape,
    with an optional chromagram.

    Parameters:
        waveform (np.ndarray): 1D audio waveform.
        sample_rate (int): Sample rate of audio.
        n_mels (int): Number of Mel bands (initial resolution, may be resized).
        n_fft (int): FFT window size.
        window_length (int): Window size for STFT.
        hop_length (int): Hop length for STFT.
        fmin (float): Minimum frequency for Mel bands.
        fmax (float): Maximum frequency for Mel bands.
        duration (float): Target audio duration in seconds (for trimming).
        trim_modality (str): 'start', 'middle', 'end', or 'stretch'.
        target_shape (tuple): Output spectrogram size (H, W).
        add_chromagram (bool): Whether to add a chromagram as an additional feature.

    Returns:
        np.ndarray: Spectrogram of shape (channels, H, W).
    """
    waveform = waveform.astype(np.float32)

    # Ensure mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    max_samples = int(sample_rate * duration)

    #if trim_modality != 'stretch':
    if len(waveform) < max_samples:
        pad_width = max_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_width), mode='constant') # adding padding is important - also for chromagram and delta mfcc
    else:

        if trim_modality == 'start':
            waveform = waveform[:max_samples]
        elif trim_modality == 'middle':
            center = len(waveform) // 2
            start = max(0, center - max_samples // 2)
            waveform = waveform[start:start + max_samples]
        elif trim_modality == 'end':
            waveform = waveform[-max_samples:]
        else:
            raise ValueError(f"Unknown trim_modality: {trim_modality}")

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        win_length=window_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    # Convert to log scale
    spec = librosa.power_to_db(mel_spec, ref=np.max)


    # Normalize mel spectrogram according to configured spectrogram frontend mode.
    spec = normalize_spectrogram_values(
        spec,
        norm_modality,
    )

    # z-score normalization
    #mean = spec.mean()
    #std = spec.std() + 1e-9  # avoid division by zero
    #spec = (spec - mean) / std

    features = [spec]  # start with mel-spec

    # Handle chromagram
    if add_chromagram:

        chroma = librosa.feature.chroma_stft(
            y=waveform,##y=non_silent_frames, #y=waveform,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )


        # Keep auxiliary hand-crafted features on min-max scaling even if mel uses AudioSet mean/std.
        chroma_norm_mode = 'minmax_0_1' if norm_modality == 'audioset_mean_std' else norm_modality
        chroma = simple_minmax_norm(chroma, chroma_norm_mode)
        # Concatenate the spectrogram and chromagram
        features.append(chroma)

    # Add MFCC + delta MFCC
    if add_delta_mfcc:
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=13,   # typical MFCC size
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax
        )
        delta_mfcc = librosa.feature.delta(mfcc)

        # Normalize both
        mfcc = z_norm(mfcc, to_zero_one = True)
        delta_mfcc = z_norm(delta_mfcc, to_zero_one = True)
        # Stack MFCC and delta MFCC together
        if add_mfcc:
            features.append(mfcc)


        if add_delta_mfcc:
            features.append(delta_mfcc)

    # Concatenate all features along frequency/channel axis
    combined_spec = np.concatenate(features, axis=0)

    return combined_spec[np.newaxis, :, :]




class SpectrogramFileDataset(Dataset):
    def __init__(
        self,
        filepaths,
        labels,
        demographics_tensor,
        config,
        transform_1D=None,
        transform_2D=None,
        return_only_audio_bool=False,
        return_dummy_audio_bool=False,
    ):
        """
        Args:
            filepaths (List[str]): List of paths to .wav files.
            labels (List[int]): Corresponding class labels.
            config (dict): Config dict for spectrogram generation.
            transform_1D (callable, optional): Optional transform to be applied on the 1D waveform tensor.
            transform_2D (callable, optional): Optional transform to be applied on the 2D spectrogram tensor.
        """
        # Validate input lengths
        if len(filepaths) != len(labels):
            raise ValueError(f"Filepaths length ({len(filepaths)}) doesn't match labels length ({len(labels)})")

        if len(filepaths) != len(demographics_tensor):
            raise ValueError(f"Filepaths length ({len(filepaths)}) doesn't match demographics_tensor length ({len(demographics_tensor)})")

        self.return_only_audio = return_only_audio_bool #model_type in train_spectrogram.models_w_custom_spectrogram_creation
        self.filepaths = filepaths
        self.labels = labels
        self.config = config
        self.transform_1D = transform_1D
        self.transform_2D = transform_2D
        self.demographics_tensor = demographics_tensor
        self.return_dummy_audio = return_dummy_audio_bool # relevant for tabular_mlp - doesn't need any audio or spectrogram

    def __len__(self):
        return len(self.filepaths)

    def pad_or_trim(self, waveform, target_length, trim_modality='end'):
        if waveform.shape[0] < target_length:
            pad_size = target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size)) # its almost the same as the code for melspectrogram above
        elif waveform.shape[0] > target_length:
            if trim_modality == 'start':
                waveform = waveform[:target_length]
            elif trim_modality == 'end':
                waveform = waveform[-target_length:]
            elif trim_modality == 'middle':
                start = (waveform.shape[0] - target_length) // 2
                waveform = waveform[start:start+target_length]
        return waveform


    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Just get demographic tensor by index
        demo_tensor = self.demographics_tensor[idx]

        if self.return_dummy_audio:
            return torch.zeros(1, dtype=torch.float32), label_tensor, demo_tensor

        y, sr = librosa.load(filepath, sr=self.config["spectrogram.sample_rate"])

        noise = self.config["spectrogram.max_noise_std"]
        if noise is not None and noise > 0:
            y = add_gaussian_noise(y, max_noise_std=noise)


        if not self.return_only_audio:



            spec = create_spectrogram_from_waveform_w_config(y, self.config)
            spec = spec.squeeze(0)
            tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # shape: (1, H, W)

            if self.transform_2D:
                tensor = self.transform_2D(tensor)
        else:
            target_length = int(self.config['spectrogram.duration'] * self.config['spectrogram.sample_rate'])

            y = self.pad_or_trim(waveform=torch.tensor(y, dtype=torch.float32),
                                target_length=target_length,
                                trim_modality=self.config['spectrogram.trim_modality'])

            # Ensure y is a tensor
            if not torch.is_tensor(y):
                tensor = torch.tensor(y)
            else:
                tensor = y



            wave_norm_mode = self.config["waveform.norm"]
            if wave_norm_mode in {"minmax_0_1", "minmax_-1_1"}:
                tensor = simple_minmax_norm(tensor, wave_norm_mode, torch_tensor=True)
            elif wave_norm_mode == "audioset_mean_std":
                # Audioset mean and std are applied in Model MOdules of prettrained models
                pass
            else:
                raise ValueError(
                    f"Unsupported waveform.norm for raw waveform path: {wave_norm_mode}"
                )
        return tensor, label_tensor, demo_tensor

