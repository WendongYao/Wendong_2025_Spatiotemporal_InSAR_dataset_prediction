from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = PROJECT_ROOT / "experiments_ext"
if str(EXTENSIONS) not in sys.path:
    sys.path.insert(0, str(EXTENSIONS))

from native_pointwise_v23 import (  # noqa: E402
    CausalTCNRegressor,
    DirectSPARRegressor,
    _weighted_mean,
)


def test_direct_spar_initialization_is_exact_anchor() -> None:
    rng = np.random.default_rng(7)
    weights = rng.normal(size=12).astype(np.float32)
    bias = 0.37
    model = DirectSPARRegressor(12, weights, bias)
    history = torch.from_numpy(rng.normal(size=(9, 12)).astype(np.float32))
    with torch.no_grad():
        prediction = model(history)
        expected = torch.einsum("bt,t->b", history, model.anchor_weights) + model.anchor_bias
    assert torch.equal(prediction, expected)


def test_causal_tcn_output_shape_and_parameter_count() -> None:
    model = CausalTCNRegressor()
    output = model(torch.zeros(5, 300))
    assert output.shape == (5,)
    assert 10_000 < sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_weighted_mean_matches_manual_value() -> None:
    values = torch.tensor([1.0, 3.0, 8.0])
    weights = torch.tensor([1.0, 2.0, 1.0])
    assert torch.isclose(_weighted_mean(values, weights), torch.tensor(3.75))
