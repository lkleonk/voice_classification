# set cache path
import torch
import torch.hub
import torch.nn as nn

from src.model.pretrained.checkpoint_utils import require_local_checkpoint
import src.utils.toolkit.cuda_handling as cuda_handling
from src.settings import PATHS



class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)

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


class VGGish(VGG):
    def __init__(self):#, pretrained=True, preprocess=True, postprocess=True, progress=True):
        super().__init__(make_layers())
        #if pretrained:
        #    state_dict = torch.load_state_dict_from_url(model_path, progress=progress)
        #    super().load_state_dict(state_dict)

    def forward(self, x): #, fs=None):
        x = VGG.forward(self, x)
        return x


class VGGishEmbedder(nn.Module):
    def __init__(self):
        super().__init__()


        full_model = VGGish()
        # Manually load model weights
        vggish_model_path = require_local_checkpoint(
            PATHS.VGGISH_WEIGHTS_PATH,
            model_name="vggish",
            source_hint="See README.md for the pinned checkpoint filename",
        )
        state_dict = torch.load(vggish_model_path)
        full_model.load_state_dict(state_dict)

        self.features = full_model.features  # conv + pooling layers
        self.embeddings = full_model.embeddings  # 128-d layer, right before classification


    @staticmethod
    def _vggish_norm(x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, 96, 64]  (log-mel patches)

        Applies VGGish-style normalization:
        log -> clamp -> shift
        """
        # If input is already log-mel, skip log.
        # If unsure, keep log — depends on your pipeline.

        x = torch.log(x + 1e-6)

        x = torch.clamp(x, min=-10.0, max=10.0)
        x = (x + 5.0) / 5.0

        return x


    def forward(self, x):

        x = self.features(x)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.embeddings(x)
        return x
