import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from src.model.pretrained.checkpoint_utils import require_local_checkpoint
from src.model.pretrained.pretrained_models_params import get_pretrained_model_spec
from src.settings import PATHS
import src.utils.toolkit.cuda_handling as cuda_handling

device = cuda_handling.set_cuda_to_gpu_nr()

_PANN_SPEC = get_pretrained_model_spec("pann")


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)





class PANNEmbedder(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = int(_PANN_SPEC["spectrogram.sample_rate"]),
        window_size: int = int(_PANN_SPEC["spectrogram.window_length"]),
        hop_size: int = int(_PANN_SPEC["spectrogram.hop_length"]),
        mel_bins: int = int(_PANN_SPEC["spectrogram.n_mels"]),
        fmin: float = float(_PANN_SPEC["spectrogram.fmin"]),
        fmax: float = float(_PANN_SPEC["spectrogram.fmax"]),
    ): # to download wget "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1" -O Cnn14_16k_mAP=0.438.pth
        super().__init__()
        #weights_path = "model_weights/Cnn14_16k_mAP=0.438.pth"

        weights_path = require_local_checkpoint(
            PATHS.PANN_WEIGHTS_PATH,
            model_name="pann",
            source_hint="See README.md for the pinned checkpoint filename",
        )
        checkpoint = torch.load(weights_path, map_location=device)


        print("=== CHECKPOINT DEBUG INFO ===")
        print("All keys:", checkpoint.keys())

        # EXTRACT THE MODEL WEIGHTS FROM THE 'model' KEY
        state_dict = checkpoint['model']
        print("PANN Model weights extracted successfully!")
        print("Number of weight tensors:", len(state_dict))
        print("First few keys:", list(state_dict.keys())[:5])

        # Filter out the spectrogram/logmel weights you don't need
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # Keep only conv, bn, fc weights (skip spectrogram/logmel)
            if not key.startswith('spectrogram_extractor') and not key.startswith('logmel_extractor'):
                filtered_state_dict[key] = value
        
        print(f"Filtered weights remaining: {len(filtered_state_dict)}/{len(state_dict)} layers")
        
        self.model = Cnn14(
            sample_rate=sample_rate, window_size=window_size, hop_size=hop_size,
            mel_bins=mel_bins, fmin=fmin, fmax=fmax, classes_num=527
        )   # Load weights
        state_dict = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(filtered_state_dict, strict=False)
        # Move to device
        #self.model = torch.hub.load('qiuqiangkong/panns_embedding', 'Cnn14', pretrained=True)

        self.model.to(device)
        #self.model = torch.hub.load('qiuqiangkong/panns_embedding', 'Cnn14', pretrained=pretrained)

    def forward(self, x):
        with torch.no_grad():
            out = self.model(x)   # dict with 'embedding' and 'clipwise_output'
        return out['embedding']   # 2048-d embedding vector
    




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


########################################
# 🔹 Global audio preprocessing params
########################################
sample_rate = int(_PANN_SPEC["spectrogram.sample_rate"])
window_size = int(_PANN_SPEC["spectrogram.window_length"])
hop_size = int(_PANN_SPEC["spectrogram.hop_length"])
mel_bins = int(_PANN_SPEC["spectrogram.n_mels"])
fmin = float(_PANN_SPEC["spectrogram.fmin"])
fmax = float(_PANN_SPEC["spectrogram.fmax"])

window = 'hann'
center = True
pad_mode = 'reflect'
ref = 1.0
amin = 1e-10
top_db = 80.0


# Spectrogram extractor
lone_spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
    win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
    freeze_parameters=True)

# Logmel feature extractor
lone_logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
    n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
    freeze_parameters=True)

# Spec augmenter
lone_spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
    freq_drop_width=8, freq_stripes_num=2)

def for_pann_to_mel_spec(x, w_augment = True):
    x = lone_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
    x = lone_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

    if w_augment:
        x = lone_spec_augmenter(x)
    return x


class Cnn14(nn.Module):
    def __init__(self, 
            sample_rate=sample_rate, 
            window_size=window_size, 
            hop_size=hop_size,
            mel_bins=mel_bins, 
            fmin=fmin, 
            fmax=fmax, 
            classes_num=527  # This should match the pre-trained model
        ):
        
        super(Cnn14, self).__init__()
        


        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def to_mel_spec(self, x):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        return x

    def forward(self, x, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x = self.to_mel_spec(x) # (batch_size, 1, time_steps, mel_bins)     
            
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        #if self.training and mixup_lambda is not None:
            #x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict






