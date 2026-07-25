from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_ext"))

from audit_tiles import TileEntry, _assert_allowed  # noqa: E402


def test_rejects_rsase_future_work_tile() -> None:
    entry = TileEntry(
        tile="E30N34",
        role="multiregion_candidate",
        status="allowed",
        path=Path("does-not-matter.csv"),
        boundary_note="",
    )
    with pytest.raises(ValueError, match="reserved for RSASE"):
        _assert_allowed(entry)


def test_original_cageo_tile_remains_allowed(tmp_path: Path) -> None:
    path = tmp_path / "E32N34.csv"
    path.touch()
    entry = TileEntry(
        tile="E32N34",
        role="original_cageo_primary",
        status="allowed",
        path=path,
        boundary_note="",
    )
    _assert_allowed(entry)
