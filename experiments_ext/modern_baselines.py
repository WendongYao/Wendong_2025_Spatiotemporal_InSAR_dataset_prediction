"""CAGEO-only modern spatiotemporal baselines.

These implementations are independent of the RSASE Transformer/STGCN line.
They adapt video-prediction mechanisms to the paper's 300-to-1 patch-residual
task and are labelled ``style`` to avoid implying an unchanged official model.
"""

from __future__ import annotations


class PatchSimVPStyleResidualModel:
    """SimVP-style encoder/translator/decoder for a single residual frame."""

    def __init__(self, input_channels: int, hidden_channels: int = 32, temporal_bins: int = 16) -> None:
        import torch.nn as nn

        class GatedSpatialBlock(nn.Module):
            def __init__(self, channels: int, kernel_size: int) -> None:
                super().__init__()
                padding = kernel_size // 2
                self.norm = nn.GroupNorm(num_groups=max(1, min(8, channels)), num_channels=channels)
                self.depthwise = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
                self.gate = nn.Conv2d(channels, channels * 2, kernel_size=1)
                self.project = nn.Conv2d(channels, channels, kernel_size=1)

            def forward(self, x):
                import torch

                residual = x
                x = self.depthwise(self.norm(x))
                value, gate = self.gate(x).chunk(2, dim=1)
                return residual + self.project(value * torch.sigmoid(gate))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_channels = hidden_channels
                self.temporal_bins = temporal_bins
                self.spatial_encoder = nn.Sequential(
                    nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(max(1, min(8, hidden_channels)), hidden_channels),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                mixed_channels = hidden_channels * temporal_bins
                translator_channels = hidden_channels * 4
                self.temporal_reduce = nn.Conv2d(mixed_channels, translator_channels, kernel_size=1)
                self.translator = nn.Sequential(
                    GatedSpatialBlock(translator_channels, 7),
                    GatedSpatialBlock(translator_channels, 11),
                    GatedSpatialBlock(translator_channels, 7),
                    GatedSpatialBlock(translator_channels, 11),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(translator_channels, hidden_channels * 2, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
                )

            def forward(self, x):
                import torch.nn.functional as F

                batch, time_steps, channels, height, width = x.shape
                encoded = self.spatial_encoder(x.reshape(batch * time_steps, channels, height, width))
                enc_h, enc_w = encoded.shape[-2:]
                encoded = encoded.reshape(batch, time_steps, self.hidden_channels, enc_h, enc_w)
                encoded = encoded.permute(0, 2, 1, 3, 4)
                binned = F.adaptive_avg_pool3d(encoded, (self.temporal_bins, enc_h, enc_w))
                mixed = binned.reshape(batch, self.hidden_channels * self.temporal_bins, enc_h, enc_w)
                translated = self.translator(self.temporal_reduce(mixed))
                return self.decoder(translated)

        self.model_class = Model

    def build(self):
        return self.model_class()
