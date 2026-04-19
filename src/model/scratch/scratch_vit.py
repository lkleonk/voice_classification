import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_utils import (
    _create_final_mlp,
    _verify_demographic_inputs,
    str_w_nrs_turn_to_int_list,
)

_DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE = 32


class ViT(nn.Module):
    def __init__(
        self,
        num_classes,
        timesteps,
        freq_bins,
        config,
        add_demographic_data,
        demographic_data_tensor_len=None,
    ):
        super().__init__()

        self.freq_bins = freq_bins
        self.timesteps = timesteps
        self.add_demographic_data = add_demographic_data
        self.add_acoustic_features = config["features.add_acoustic_feature"]
        self.extra_demographic_data_mlp = config["features.extra_demographic_data_mlp"]
        self.early_fusion_add_extra_data = config["vit.early_fusion_add_extra_data"]
        self.late_fusion_add_extra_data = config["vit.late_fusion_add_extra_data"]
        self.demographic_data_tensor_len = demographic_data_tensor_len

        _verify_demographic_inputs(self.add_demographic_data, self.demographic_data_tensor_len)
        demo_len = self.demographic_data_tensor_len
        if self.add_demographic_data:
            if demo_len is None:
                raise ValueError("demographic_data_tensor_len must be set when add_demographic_data=True")
        else:
            demo_len = 0
        assert demo_len is not None

        if self.add_demographic_data:
            if not self.early_fusion_add_extra_data and not self.late_fusion_add_extra_data:
                raise ValueError(
                    "When add-demographic_data is activated, at least one of the two following "
                    "variables have to be activated: early_fusion_add_extra_data and late_fusion_add_extra_data"
                )

        if self.extra_demographic_data_mlp and not self.late_fusion_add_extra_data:
            print(
                "When extra_demographic_data_mlp is activated, late_fusion_add_extra_data "
                "should not be activated. Please change accordingly"
            )

        if self.add_acoustic_features:
            assert self.add_demographic_data, (
                "if add_acoustic_features is activated, add_demogrpahic_data also has to be activated"
            )

        self.patch_size = config["vit.patch_size"]
        emb_dim = config["vit.emb_dim"]
        depth = config["vit.depth"]
        num_heads = config["vit.num_heads"]
        mlp_ratio = config["vit.mlp_ratio"]
        layer_dropout = config["vit.layer_dropout"]
        final_dropout = config["vit.final_dropout"]
        mlp_layer_sizes = str_w_nrs_turn_to_int_list(config["vit.mlp_layer_sizes"])

        self.in_chans = 1
        height, width = freq_bins, timesteps
        extra_feature_rows = 0
        if config["spectrogram.add_chromagram"]:
            extra_feature_rows += 12
        if config["spectrogram.add_mfcc"]:
            extra_feature_rows += 13
        if config["spectrogram.add_delta_mfcc"]:
            extra_feature_rows += 13

        if height % self.patch_size != 0:
            raise ValueError(
                "ViT requires the effective spectrogram height to be divisible by "
                f"vit.patch_size={self.patch_size}, but got height={height}. "
                f"Current input shape is ({height}, {width}). "
                f"Height is derived from spectrogram.n_mels plus optional extra feature rows "
                f"(current extra rows: {extra_feature_rows}). "
                "Choose spectrogram.n_mels so the effective height is divisible by the patch "
                "size, for example an Optuna/search range like "
                "'spectrogram.n_mels': ['int', 128, 256, 16]."
            )

        self.padded_width = width
        if width % self.patch_size != 0:
            self.padded_width = ((width + self.patch_size - 1) // self.patch_size) * self.patch_size
            warnings.warn(
                "ViT input width is not divisible by vit.patch_size="
                f"{self.patch_size}. Width will be zero-padded from {width} to "
                f"{self.padded_width} before patch extraction.",
                stacklevel=2,
            )

        num_patches = (height // self.patch_size) * (self.padded_width // self.patch_size)
        if self.early_fusion_add_extra_data:
            num_patches += 1
        patch_dim = self.in_chans * self.patch_size * self.patch_size

        self.patch_embed = nn.Linear(patch_dim, emb_dim)
        if self.early_fusion_add_extra_data and self.add_demographic_data:
            self.extra_data_to_token = nn.Linear(demo_len, emb_dim)
        else:
            self.extra_data_to_token = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            dropout=layer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        if self.add_demographic_data and self.extra_demographic_data_mlp:
            self.demo_mlp = nn.Sequential(
                nn.Linear(demo_len, _DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

        mlp_input_size = emb_dim
        if self.add_demographic_data and demo_len:
            if self.extra_demographic_data_mlp:
                mlp_input_size += _DEMOGRAPHIC_DATA_MLP_OUTPUT_SIZE
            else:
                mlp_input_size += demo_len

        self.mlp_head = _create_final_mlp(
            mlp_input_size, mlp_layer_sizes, output=num_classes, dropout=final_dropout
        )

    def forward(self, x, demo_tensor=None):
        batch_size, channels, _, _ = x.shape
        if channels != self.in_chans:
            raise ValueError(
                f"ViT was initialized for {self.in_chans} channel(s) but got input with {channels} channels."
            )

        width_pad = self.padded_width - x.shape[-1]
        if width_pad < 0:
            raise ValueError(
                f"ViT received input width {x.shape[-1]}, which exceeds the configured width "
                f"{self.padded_width} after padding setup."
            )
        if width_pad > 0:
            x = F.pad(x, (0, width_pad, 0, 0))

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(
            batch_size, channels, -1, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(batch_size, -1, channels * self.patch_size * self.patch_size)

        x = self.patch_embed(patches)

        if self.early_fusion_add_extra_data and self.add_demographic_data:
            if demo_tensor is None:
                raise ValueError("demographic data must be provided when add_demographic_data=True")
            assert self.extra_data_to_token is not None
            ef_token = self.extra_data_to_token(demo_tensor).unsqueeze(1)
            x = torch.cat((ef_token, x), dim=1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)

        cls_rep = x[:, 0, :]
        if self.add_demographic_data and demo_tensor is not None:
            if self.extra_demographic_data_mlp:
                demo_tensor = self.demo_mlp(demo_tensor)
            cls_rep = torch.cat((cls_rep, demo_tensor), dim=1)

        return self.mlp_head(cls_rep)

