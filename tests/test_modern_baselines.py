from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_ext"))

from modern_baselines import PatchSimVPStyleResidualModel  # noqa: E402


def test_simvp_style_shape() -> None:
    model = PatchSimVPStyleResidualModel(input_channels=2, hidden_channels=16, temporal_bins=4).build()
    x = torch.randn(2, 12, 2, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 16, 16)
