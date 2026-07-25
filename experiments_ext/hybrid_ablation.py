"""Controlled component removals for the CAGEO Hybrid CNN-LSTM.

The clean public implementation remains untouched in ``source/``.  These
helpers disable complete additive branches after the public builder creates a
model, which keeps the remaining weights and training protocol identical.
"""

from __future__ import annotations


def apply_hybrid_ablation(
    model,
    *,
    disable_recent_gate: bool = False,
    disable_spatial_correction: bool = False,
) -> tuple[str, ...]:
    """Disable requested additive Hybrid branches and freeze their weights."""

    import torch

    disabled: list[str] = []
    with torch.no_grad():
        if disable_recent_gate:
            if not hasattr(model, "recent_scale") or not hasattr(model, "recent_gate"):
                raise TypeError("Recent-gate ablation requires a Hybrid model.")
            model.recent_scale.zero_()
            model.recent_scale.requires_grad_(False)
            for parameter in model.recent_gate.parameters():
                parameter.requires_grad_(False)
            disabled.append("recent_gate")

        if disable_spatial_correction:
            required = ("correction_scale", "frame_encoder", "cells", "decoder", "correction_head")
            if any(not hasattr(model, name) for name in required):
                raise TypeError("Spatial-correction ablation requires a Hybrid CNN-LSTM model.")
            model.correction_scale.zero_()
            model.correction_scale.requires_grad_(False)
            for module_name in ("frame_encoder", "cells", "decoder", "correction_head"):
                for parameter in getattr(model, module_name).parameters():
                    parameter.requires_grad_(False)
            disabled.append("spatial_correction")
    return tuple(disabled)


def trainable_parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
