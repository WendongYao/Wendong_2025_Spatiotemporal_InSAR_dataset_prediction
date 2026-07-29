from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_ext"))

from native_support_baselines import DLinear  # noqa: E402


def test_dlinear_shape_and_parameter_count() -> None:
    import torch

    model = DLinear.build(sequence_length=300, moving_average=25)
    values = torch.randn(7, 300)
    output = model(values)
    assert output.shape == (7,)
    assert sum(parameter.numel() for parameter in model.parameters()) == 602


def test_dlinear_moving_mean_preserves_constant_history() -> None:
    import torch

    model = DLinear.build(sequence_length=15, moving_average=5)
    values = torch.full((3, 15), 2.5)
    moving = model.moving_mean(values)
    assert np.allclose(moving.detach().numpy(), 2.5)
