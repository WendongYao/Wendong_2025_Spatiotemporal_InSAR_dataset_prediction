"""Lightweight support-aware point-query increment model for CAGEO.

The model keeps the CAGEO patch context, but queries a prediction at each raw
measurement (or arbitrary grid coordinate) using the point's direct 300-step
history. A direct raw-LASSO forecast is transformed into standardized
future-increment coordinates and used as a fixed anchor; the learned network
only predicts a support-conditioned correction in those same coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupportAwarePointQueryModel(nn.Module):
    """Point-query residual predictor with an optional fixed LASSO anchor."""

    def __init__(
        self,
        *,
        input_channels: int,
        time_steps: int,
        patch_size: int,
        anchor_weights: torch.Tensor | None,
        anchor_bias: float = 0.0,
        context_frames: int = 16,
        temporal_channels: int = 24,
        spatial_channels: int = 32,
        use_spatial_context: bool = True,
        use_global_coordinates: bool = True,
        use_local_coordinates: bool = True,
    ) -> None:
        super().__init__()
        if context_frames < 1 or context_frames > time_steps:
            raise ValueError("context_frames must be within the available history.")
        self.time_steps = int(time_steps)
        self.context_frames = int(context_frames)
        self.patch_size = int(patch_size)
        self.input_channels = int(input_channels)
        self.use_spatial_context = bool(use_spatial_context)
        self.use_global_coordinates = bool(use_global_coordinates)
        self.use_local_coordinates = bool(use_local_coordinates)
        # Preserve the complete measurement history. Global temporal pooling
        # erased recent evolution that the linear anchor could not represent.
        self.temporal_encoder = nn.Sequential(
            nn.Linear(time_steps, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, temporal_channels),
            nn.GELU(),
        )
        self.context_encoder = (
            nn.Sequential(
                nn.Conv2d(input_channels * context_frames, spatial_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(spatial_channels, spatial_channels, kernel_size=3, padding=1),
                nn.GELU(),
            )
            if self.use_spatial_context
            else None
        )
        decoder_input = temporal_channels
        if self.use_local_coordinates:
            decoder_input += 2
        if self.use_spatial_context:
            decoder_input += spatial_channels
        if self.use_global_coordinates:
            decoder_input += 2
        self.query_decoder = nn.Sequential(
            nn.Linear(decoder_input, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )
        # Preserve the fitted raw-LASSO anchor at initialization.  A random
        # residual head would perturb a strong linear baseline before the
        # support-aware correction has learned anything; zero initialization
        # makes the first forward pass exactly the anchor prediction.
        nn.init.zeros_(self.query_decoder[-1].weight)
        nn.init.zeros_(self.query_decoder[-1].bias)
        self.anchor_preserving_initialization = True
        self.correction_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        if anchor_weights is None:
            self.register_buffer("anchor_weights", torch.zeros(time_steps, dtype=torch.float32))
            self.anchor_enabled = False
        else:
            weights = anchor_weights.detach().float().reshape(time_steps)
            self.register_buffer("anchor_weights", weights)
            self.anchor_enabled = True
        self.register_buffer("anchor_bias", torch.tensor(float(anchor_bias), dtype=torch.float32))

    def _anchor(self, query_history: torch.Tensor) -> torch.Tensor:
        if not self.anchor_enabled:
            return torch.zeros(
                query_history.shape[0], query_history.shape[1],
                device=query_history.device, dtype=query_history.dtype,
            )
        return torch.einsum("bnt,t->bn", query_history, self.anchor_weights) + self.anchor_bias

    def forward(
        self,
        context_inputs: torch.Tensor,
        query_coordinates: torch.Tensor,
        query_history: torch.Tensor,
        global_query_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized residual values with shape ``[B, N]``."""
        if context_inputs.ndim != 5:
            raise ValueError("context_inputs must have shape [B,T,C,H,W].")
        if query_coordinates.ndim != 3 or query_coordinates.shape[-1] != 2:
            raise ValueError("query_coordinates must have shape [B,N,2].")
        if query_history.ndim != 3 or query_history.shape[-1] != self.time_steps:
            raise ValueError("query_history must have shape [B,N,T].")
        if global_query_coordinates is None:
            global_query_coordinates = query_coordinates
        if global_query_coordinates.ndim != 3 or global_query_coordinates.shape[-1] != 2:
            raise ValueError("global_query_coordinates must have shape [B,N,2].")
        batch_size, _, _, height, width = context_inputs.shape
        if (
            query_coordinates.shape[0] != batch_size
            or query_history.shape[:2] != query_coordinates.shape[:2]
            or global_query_coordinates.shape[:2] != query_coordinates.shape[:2]
        ):
            raise ValueError("Query batch and point dimensions do not match.")

        point_count = query_history.shape[1]
        temporal = self.temporal_encoder(query_history.reshape(batch_size * point_count, self.time_steps))
        temporal = temporal.reshape(batch_size, point_count, -1)
        decoder_features = [temporal]
        if self.use_local_coordinates:
            decoder_features.append(query_coordinates)
        if self.use_spatial_context:
            recent = context_inputs[:, -self.context_frames :]
            recent = recent.reshape(batch_size, self.context_frames * self.input_channels, height, width)
            context_features = self.context_encoder(recent)
            sampled_context = F.grid_sample(
                context_features,
                query_coordinates[:, None, :, :],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[:, :, 0, :].transpose(1, 2)
            decoder_features.append(sampled_context)
        if self.use_global_coordinates:
            decoder_features.append(global_query_coordinates)
        decoded = self.query_decoder(torch.cat(decoder_features, dim=-1)).squeeze(-1)
        return self._anchor(query_history) + self.correction_scale * decoded
