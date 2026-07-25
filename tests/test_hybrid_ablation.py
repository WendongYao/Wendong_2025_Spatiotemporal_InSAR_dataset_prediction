from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_ext"))

from hybrid_ablation import apply_hybrid_ablation, trainable_parameter_count  # noqa: E402


def test_hybrid_branches_can_be_disabled_and_frozen() -> None:
    import torch
    import torch.nn as nn

    class TinyHybrid(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recent_scale = nn.Parameter(torch.tensor(1.0))
            self.correction_scale = nn.Parameter(torch.tensor(0.1))
            self.recent_gate = nn.Conv2d(1, 1, 1)
            self.frame_encoder = nn.Conv2d(1, 1, 1)
            self.cells = nn.ModuleList([nn.Conv2d(1, 1, 1)])
            self.decoder = nn.Conv2d(1, 1, 1)
            self.correction_head = nn.Conv2d(1, 1, 1)
            self.linear_head = nn.Conv2d(1, 1, 1)

    model = TinyHybrid()
    before = trainable_parameter_count(model)
    disabled = apply_hybrid_ablation(model, disable_recent_gate=True, disable_spatial_correction=True)
    assert disabled == ("recent_gate", "spatial_correction")
    assert model.recent_scale.item() == 0.0
    assert model.correction_scale.item() == 0.0
    assert not model.recent_scale.requires_grad
    assert not model.correction_scale.requires_grad
    assert trainable_parameter_count(model) < before
    assert all(not p.requires_grad for p in model.recent_gate.parameters())
    assert all(not p.requires_grad for p in model.correction_head.parameters())
