"""What each heavy stage's per-case result depends on, as a key.

Most cases share their register rows
with another case (the financing step accounts once per register key and tiles
the result to every case on the key), and most stages read only a few levers
beyond those rows.  A stage that computes one result per distinct *fingerprint*
and hands a relabelled copy to every other case on it gives the numbers it gave
when it computed every case, provided the fingerprint holds everything the
stage reads.  This module is where those fingerprints are written down, so the
stages and the lever-read audit test (``tests/unit/test_f20_phase2_fingerprints.py``)
read one definition.

**The register body** (:func:`body_key`).  The financing step
(:func:`calculate_financing_scenarios.calculate_financing_scenarios`) builds each
register key's rows once and tiles them; afterwards only two bookings differ by
case: the LDES floor (market-bill cases only; on the case's ``ldes_floor_basis``
and its base case's floor rate, ``public_account.ldes_floor_topups``) and, in a
vintaged (2050) register, the Capacity Market design (per spec).  So two 2030
cases with the same register levers, gated levers, market-bill flag and (when
they go through the market bill) floor basis and base case have the same rows,
column for column, apart from ``case_id`` and the lever columns.  A vintaged
register shares nothing (``vintaged=True``: the key is the case itself); it has
about 16 cases, so nothing is lost.

**The stage keys** add what each stage reads beyond the rows:

* :func:`savings_key` (``cfd_repricing.per_row_savings``): the comparator, the
  strike basis and the amortisation (the pre-2025 boundary is a register lever).
* :func:`bridge_key` (the consumer-bill bridge): the comparator the financing
  summary names.
* :func:`tariff_key` (the nationalised tariff): the base case, the comparator and
  every lever the tariff reads that is not already in the body
  (:data:`TARIFF_UNREAD_LEVERS` are the ones it is known not to read).

Each stage checks itself at run time too: :class:`SampleCheck` recomputes about
2% of the cases that took a shared result (the first, then every
:data:`SAMPLE_EVERY`-th) in full and raises :class:`FingerprintMismatch` on any
difference, so a lever a stage starts to read without its key being widened
fails the run instead of moving a number.
"""

from __future__ import annotations

import os
from typing import Mapping

from scripts.public_ownership.levers import (
    AXIS_CFD_AMORTISATION, AXIS_CFD_STRIKE_BASIS, AXIS_LDES_FLOOR_BASIS, GATED_LEVERS, LEVERS, REGISTER_LEVERS,
    base_case, lever_default, spec_for_row,
)

#: Shared cases recomputed in full by :class:`SampleCheck`: the first, then one in this many (2%).
SAMPLE_EVERY = 50
#: ``PYPSA_GB_F20_SAMPLE_EVERY`` overrides :data:`SAMPLE_EVERY` (1 recomputes every shared case; 0 none).
SAMPLE_ENV = "PYPSA_GB_F20_SAMPLE_EVERY"
#: ``PYPSA_GB_F20_SHARE=0`` turns the sharing off in every stage (each case computed in full, as before F20 phase 2).
SHARE_ENV = "PYPSA_GB_F20_SHARE"

#: Levers the financing step books on the rows of a market-bill case after the key's account (the LDES floor).
BODY_BOOKED_LEVERS = (AXIS_LDES_FLOOR_BASIS,)
#: The levers :func:`cfd_repricing.per_row_savings` reads beyond the rows and the comparator.
SAVINGS_LEVERS = (AXIS_CFD_STRIKE_BASIS, AXIS_CFD_AMORTISATION)
#: Levers the nationalised tariff never reads (beyond what the register body already holds): the
#: market-bill levers of the CfD re-pricing and the public account, and the LDES floor basis (an
#: acquiring case is not on the market bill, so the financing step books it no floor).  Every other
#: lever is in :func:`tariff_key`; the lever-read audit test checks this list.
TARIFF_UNREAD_LEVERS = (
    AXIS_CFD_STRIKE_BASIS, AXIS_CFD_AMORTISATION, AXIS_LDES_FLOOR_BASIS,
)


def sharing_enabled() -> bool:
    """Whether the stages share results across cases (``PYPSA_GB_F20_SHARE``, default on)."""
    return os.environ.get(SHARE_ENV, "1").strip() not in ("0", "false", "False", "no")


def sample_every() -> int:
    """How often a shared case is recomputed in full (:data:`SAMPLE_EVERY`, or ``PYPSA_GB_F20_SAMPLE_EVERY``)."""
    text = os.environ.get(SAMPLE_ENV)
    return SAMPLE_EVERY if text is None or not text.strip() else max(0, int(text))


def register_is_keyed(path) -> bool:
    """Whether a register file carries every register lever (its settings table, or an old CSV's columns).

    The fingerprints read a case's register levers from its settings; a register without them
    (a test's hand-built frame) gives every case the default point, so nothing may be shared.
    """
    from pathlib import Path

    import pandas as pd

    from scripts.public_ownership.register_io import settings_path

    path = Path(path)
    source = settings_path(path) if path.suffix != ".csv" and settings_path(path).exists() else path
    if not source.exists():
        return False
    return set(REGISTER_LEVERS) <= set(pd.read_csv(source, nrows=0).columns)


def frame_is_keyed(frame) -> bool:
    """:func:`register_is_keyed` for a register frame (its lever columns)."""
    return set(REGISTER_LEVERS) <= set(frame.columns)


def register_vintaged(path) -> bool:
    """Whether a register file is for a vintaged (2040-on) model year, read off private's ``model_year``."""
    import pandas as pd

    from scripts.public_ownership.model_year import is_vintaged_model_year
    from scripts.public_ownership.register_io import read_register

    rows = read_register(path, columns=["case_id", "model_year"], cases=["private"])
    if "model_year" not in rows.columns:
        return False
    return any(is_vintaged_model_year(int(y)) for y in pd.to_numeric(rows["model_year"], errors="coerce").dropna().unique())


def _value(settings: Mapping[str, str], name: str) -> str:
    value = settings.get(name)
    if value is None or (isinstance(value, float) and value != value):
        return lever_default(name)
    return str(value)


def body_key(case_id: str, settings: Mapping[str, str], *, vintaged: bool = False) -> tuple:
    """Cases with the same key have the same register rows apart from ``case_id`` and the lever columns.

    ``settings`` is the case's lever point (``cfd_repricing.register_settings_by_case``:
    every lever, gated ones where the case sets them).  ``vintaged`` (a 2040-on
    register) shares nothing.
    """
    if vintaged:
        return ("case", str(case_id))
    market = spec_for_row(settings).market_bill()
    key = (tuple(_value(settings, name) for name in REGISTER_LEVERS),
           tuple(_value(settings, name) for name in GATED_LEVERS), market)
    if market:
        # The LDES floor booked on a market case's rows: its basis and its base case's floor rate.
        key += (tuple(_value(settings, name) for name in BODY_BOOKED_LEVERS), base_case(case_id))
    return key


def savings_key(case_id: str, settings: Mapping[str, str], comparator: str, *, vintaged: bool = False) -> tuple:
    """:func:`cfd_repricing.per_row_savings` for a case is a function of this key (labels apart)."""
    return (body_key(case_id, settings, vintaged=vintaged), str(comparator),
            tuple(_value(settings, name) for name in SAVINGS_LEVERS))


def bridge_key(case_id: str, settings: Mapping[str, str], comparator: str, *, vintaged: bool = False) -> tuple:
    """The bridge's technology table for a case is a function of this key (its ``case_id`` apart)."""
    return (body_key(case_id, settings, vintaged=vintaged), str(comparator))


def tariff_key(case_id: str, settings: Mapping[str, str], comparator: str, *, vintaged: bool = False,
               extra: tuple = ()) -> tuple:
    """The nationalised tariff of a case is a function of this key (its ``case_id`` and ``levers`` apart).

    ``extra``: anything else the caller's run reads per case (the carried acquisition id).
    """
    read = tuple((name, _value(settings, name)) for name in (*LEVERS, *GATED_LEVERS)
                 if name not in TARIFF_UNREAD_LEVERS and name not in REGISTER_LEVERS)
    return (body_key(case_id, settings, vintaged=vintaged), base_case(case_id), str(comparator), read, tuple(extra))


class FingerprintMismatch(RuntimeError):
    """A shared result differs from the case's own full computation: the stage's fingerprint misses something."""


class SampleCheck:
    """Which shared cases to recompute in full: the first, then every ``every``-th (F20 phase 2).

    ``check(label, case_id, shared, full)`` raises :class:`FingerprintMismatch` when
    ``same(shared, full)`` is false; ``due()`` says whether the next shared case is one
    to recompute (call it once per shared case).
    """

    def __init__(self, stage: str, every: int | None = None):
        self.stage = stage
        self.every = sample_every() if every is None else int(every)
        self.shared = 0
        self.checked = 0

    def due(self) -> bool:
        self.shared += 1
        if self.every <= 0:
            return False
        return self.shared == 1 or self.shared % self.every == 0

    def check(self, case_id: str, same: bool, detail: str = "") -> None:
        self.checked += 1
        if not same:
            raise FingerprintMismatch(
                f"{self.stage}: {case_id}'s shared result differs from its own full computation{detail}; "
                "the stage reads something its fingerprint (scripts/public_ownership/fingerprints.py) leaves out")

    def report(self) -> str:
        return f"{self.stage}: {self.shared} case(s) took a shared result, {self.checked} recomputed in full and equal"
