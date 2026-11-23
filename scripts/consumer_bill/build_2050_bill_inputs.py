"""The three 2050 bill input files: the generic long-range builder for 2050 (task Y13).

Writes

* ``data/market/consumer_bill_benchmarks/2050_pypsa_informed_assumptions.json``;
* ``data/market/consumer_bill_forecast/2050_recovery_demand_assumptions.json``;
* ``data/market/consumer_bill_forecast/2050_ro_scheme_parameters.json``.

User, 14 Nov 2026: no model figure is pinned here. This module was the 2050 builder with the Capacity Market
and Y12's network roll-forward typed in and tests holding them equal to the model; it is now the 2050 entry
point of ``build_fes_bill_inputs.py``, which reads both from their source every time it runs: the Capacity
Market from ``resources/market/capacity_market/cm_calendar_2050_summary.json`` (rule
``build_capacity_market_2050``) and the roll-forward from the network account's own function
(``network_drivers``). The generic builder reproduced this one byte for byte before the change
(``tests/unit/test_fes_bill_inputs.py``). The pinned version is in git history.

Usage::

    python -m scripts.consumer_bill.build_2050_bill_inputs [--check]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.consumer_bill import build_fes_bill_inputs as _generic  # noqa: E402

MODEL_YEAR = 2050
OUTPUTS = _generic.outputs(MODEL_YEAR)
#: Sourced inputs the generic builder holds (re-exported for the tests that read them here).
DUOS_FY2030_31_BASE_2050 = _generic.DUOS_FY2030_31_BASE_2050
NESO_INTERNAL_BSUOS_2024_25 = _generic.NESO_INTERNAL_BSUOS_2024_25
render = _generic.render


def build(root: str | Path = ".") -> dict[str, dict]:
    """The 2050 documents (the generic builder for 2050)."""
    return _generic.build(MODEL_YEAR, root)


def keep_committed_refresh(recovery: dict, path=None) -> dict:
    """The built recovery document with the committed file's refresh re-applied (the generic rule)."""
    return _generic.keep_committed_refresh(recovery, path if path is not None else OUTPUTS["recovery"])


def bsuos_rebase_2050(root) -> dict:
    """Decision 65's BSUoS rebase on the 2050 ancillary table (the generic builder's)."""
    return _generic.bsuos_rebase_2050(root)


def cm_2050(root: str | Path = ".") -> dict:
    """The 2050 Capacity Market figures the bill carries, read from the auction model's output."""
    return _generic.cm_figures(MODEL_YEAR, root)


def network_2050(root: str | Path = ".") -> dict:
    """Y12's 2050 roll-forward, from the network account's own function."""
    return _generic.network_drivers(MODEL_YEAR, root)


def y12_driver(sector: str, year: int, root: str | Path = ".") -> float:
    """Y12's revenue driver in ``year`` against 2030, on the network account's roll-forward."""
    return _generic.y12_driver(network_2050(root), sector, year)


def main(argv=None) -> None:
    _generic.main(["--year", str(MODEL_YEAR), *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
