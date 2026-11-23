"""The 2050 support contracts as one book, settled through one path (task SETTLE2050).

**What was missing.** The vintaged 2050 register gives each build-year cohort the
contract it holds in the model year (Y10: ``label_support_contracts``; Y11's known
AR7 phases and HPC, ``PAR7_<build year>`` CfDs, ``DPA_<build year>`` agreements), but
nothing settled those labels: the bill, the balancing-cost extraction and the asset
settlement all read the support labels of the *network*, and the HT50 build carries
none. So on the gate week the CfD line, ``cfd_support_gbp`` and ``dpa_support_gbp``
were 0 on 164 GW of CfD and 38 GW of DPA plant (plan Y14, E2E block).

**The book.** :func:`build_contract_book` turns the register's labelled rows into one
row per *contract unit* (a register row holding a live contract), with the network
asset it is a share of (``settlement_share``: the vintaged cohort share times the
site share, :func:`settlement.vintaged_settlement_share`), the share of the year the
contract is live (``live_share``) and its terms. Every 2050 stage settles from it:

* **CfDs** (AR7 phases, HPC, ``PAR7_<year>`` renewables, SMRs and BECCS):
  :func:`cfd_units` expands the network's dispatch into contract units (asset
  dispatch x ``settlement_share`` x ``live_share``), which go through
  ``revenue_tracking.compute_cfd_payments`` exactly as a labelled 2030 network's CfD
  generators do: in the bill (``project_future_bill``), in the paid-energy probe of
  the strike adjustments, in the constraint line's curtailment compensation
  (``extract_future_balancing_costs``) and in the asset settlement. The strike table
  is :func:`strike_table_frame` (Y11's strikes in the published table's format, so
  the 2030 loader, the round resolution and route R1's per-key cuts are unchanged).
* **DPAs** (gas CCS generators and hydrogen-turbine links): the settlement sums each
  network asset's energy terms over the year (:func:`dpa_energy_terms`) and
  :func:`settle_dpa_book` applies each unit's contract through
  ``support_regime_2050.dpa_payments`` (availability + variable payment -
  gainshare). The bill's ``dpa`` line reads the settlement's total
  (``model_linked_rules.dpa``), so the bill and the register carry the same money.

The book is a 2050 artefact: no 2030 stage is handed one, and every function here
is inert without it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

#: Columns of the book file, in order.
BOOK_COLUMNS = (
    "unit", "network_asset", "component_type", "carrier", "stock_flow_technology", "build_year", "capacity_mw",
    "scheme", "cfd_round", "cfd_contract_id", "strike_key", "strike_gbp_per_mwh_2024", "settlement_share",
    "live_share", "energy_factor", "dpa_carrier", "dpa_technology", "dpa_apr_gbp_per_kw_year",
    "dpa_expected_availability", "dpa_term_years", "dpa_margin_gbp_per_mwh", "dpa_gainshare_share",
    "dpa_gainshare_threshold_gbp_per_kw_year", "dpa_private_rate", "dpa_fixed_cost_gbp_per_kw_year", "source",
)
SCHEME_CFD = "CfD"
SCHEME_DPA = "DPA"
#: Register support labels the book settles (Sizewell C's RAB is an external bill line; LDES
#: is settled by the cap-and-floor route, item D4(c)).
BOOK_SCHEMES = (SCHEME_CFD, SCHEME_DPA)
#: Network carrier -> the DPA scheme carrier of ``support_regime_2050.json``.
DPA_CARRIERS = {"CCS Gas": "CCS Gas", "H2_turbine": "Hydrogen"}
#: Vintage-table technology -> the published strike table's technology name. A key the 2050
#: table settles must be reachable from the unit's network carrier
#: (``cfd_strike_table.carrier_to_technology``), so SMRs and large nuclear share ``nuclear``.
TECHNOLOGY_TABLE_NAME = {
    "Offshore wind": "offshore", "Floating offshore wind": "offshore", "Onshore wind": "onshore",
    "Solar PV": "solar", "Tidal stream": "tidal", "Small modular reactors (SMRs)": "nuclear",
    "Large-scale nuclear": "nuclear", "Hydropower": "hydro", "Biomass - with CCUS - mature": "beccs",
    # Year-generic stage 5b: the live AR4 energy-from-waste and AR5 geothermal contracts of a 2040 year.
    "Energy from waste": "energy_from_waste", "Geothermal": "geothermal", "Biomass - unabated": "dedicated_biomass",
}
#: Technologies whose post-AR7 (``PAR7_<year>``) strikes the 2050 table carries: the register's
#: CfD technologies (``fleet_vintages.CFD_TECHNOLOGIES``) that Y11 prices through
#: ``levelised_strike``. The stock-flow fleet has one offshore line, so floating offshore is
#: not a separate vintage (its strikes would collide with fixed offshore on ``offshore``);
#: hydro is merchant; large nuclear holds HPC and Sizewell C only; BECCS comes from
#: ``beccs_strike``.
PAR7_TECHNOLOGIES = ("Offshore wind", "Onshore wind", "Solar PV", "Tidal stream", "Small modular reactors (SMRs)")
#: Columns a register must carry to be booked.
REGISTER_COLUMNS = ("asset_id", "component_type", "carrier", "support_type", "cfd_round", "cfd_live_share",
                    "capacity_mw", "build_year", "stock_flow_technology", "cohort_share")
TOLERANCE_GBP = 1.0


class ContractBookError(ValueError):
    """A book that cannot be built, read or settled."""


# ── the strike table ─────────────────────────────────────────────────────────

def strike_table_frame(strikes: pd.DataFrame, known: pd.DataFrame, *, variant: str = "central",
                       beccs: pd.DataFrame | None = None, cpi_2012_to_2024: float | None = None) -> pd.DataFrame:
    """Y11's 2050 strikes in the published strike table's format (2012 prices, one row per entry).

    * ``PAR7_<year>`` rows: :func:`cfd_strikes_2050.strike_table_2050` (``variant``) for the
      technologies in :data:`TECHNOLOGY_TABLE_NAME` and, when given, the BECCS strikes
      (``support_regime_2050.beccs_strike`` rows, technology ``beccs``).
    * Known contracts live in the model year (``cfd_strikes_2050.known_contracts_2050``):
      AR7 offshore, floating and Scottish units, AR7a and HPC, each weighted by its live MW.
      ``load_strike_table`` collapses a round's rows to a capacity-weighted mean per
      ``(round, technology)``, exactly as the 2030 table collapses delivery years, so
      AR7's English (91.20), Scottish (89.49) and floating (216.49) strikes settle as one
      ``(AR7, offshore)`` entry: a network unit's carrier cannot tell them apart.

    Args:
        strikes: ``strike_table_2050`` output.
        known: ``known_contracts_2050`` output.
        variant: Strike variant (``central``, ``neso_wind_ramp``, ``learning``, ...).
        beccs: BECCS strike rows (``build_year``, ``strike_gbp_per_mwh_2024``).
        cpi_2012_to_2024: The 2012-to-2024 factor; ``cfd_strikes_2050.CPI_2012_TO_2024`` if None.

    Returns:
        ``cfd_round, technology, delivery_year, strike_price_gbp_per_mwh_2012, capacity_mw,
        price_basis, source, notes``.
    """
    from scripts.market import cfd_strikes_2050 as cfd50

    factor = float(cfd50.CPI_2012_TO_2024 if cpi_2012_to_2024 is None else cpi_2012_to_2024)
    rows = []
    chosen = strikes[strikes["variant"].eq(variant)]
    if chosen.empty:
        raise ContractBookError(f"no strikes for variant {variant!r}")
    for _, r in chosen.iterrows():
        if str(r["technology"]) not in PAR7_TECHNOLOGIES:
            continue
        name = TECHNOLOGY_TABLE_NAME[str(r["technology"])]
        year = int(r["build_year"])
        rows.append({"cfd_round": f"{cfd50.MODELLED_ROUND_PREFIX}{year}", "technology": name,
                     "delivery_year": f"{year}-{str(year + 1)[-2:]}",
                     "strike_price_gbp_per_mwh_2012": float(r["strike_gbp_per_mwh_2024"]) / factor,
                     "capacity_mw": 1.0, "price_basis": "2012 prices",
                     "source": "scripts/market/cfd_strikes_2050.py levelised_strike (Y11, decision 6)",
                     "notes": f"{r['technology']} {year}, variant {variant}: {r['strike_gbp_per_mwh_2024']:.4f} GBP/MWh "
                              f"in 2024 prices at the private rate {float(r['private_rate']):.4f}"})
    for _, r in (beccs if beccs is not None else pd.DataFrame()).iterrows():
        year = int(r["build_year"])
        rows.append({"cfd_round": f"{cfd50.MODELLED_ROUND_PREFIX}{year}", "technology": "beccs",
                     "delivery_year": f"{year}-{str(year + 1)[-2:]}",
                     "strike_price_gbp_per_mwh_2012": float(r["strike_gbp_per_mwh_2024"]) / factor,
                     "capacity_mw": 1.0, "price_basis": "2012 prices",
                     "source": "scripts/public_ownership/support_regime_2050.py beccs_strike (Y11)",
                     "notes": f"Power BECCS {year}, electricity leg after the off-bill removal credit"})
    # User decision (13 Nov 2026): the 2029-30 cohort's modelled AR8 CfD (fleet_vintages.AR8_MODELLED_ROUND) at the
    # 2030 chain's own strikes (the published table's AR8_modelled rows), one per technology the cohort holds.
    from scripts.market.cfd_strikes import PUBLISHED_STRIKES
    from scripts.public_ownership.fleet_vintages import AR8_MODELLED_CFD_SHARE, AR8_MODELLED_ROUND

    published = pd.read_csv(PUBLISHED_STRIKES)
    for name in sorted({TECHNOLOGY_TABLE_NAME[t] for t in AR8_MODELLED_CFD_SHARE}):
        hit = published[published["cfd_round"].eq(AR8_MODELLED_ROUND) & published["technology"].eq(name)]
        if len(hit) != 1:
            raise ContractBookError(f"the published strike table has {len(hit)} {AR8_MODELLED_ROUND} {name} rows")
        r = hit.iloc[0]
        rows.append({"cfd_round": AR8_MODELLED_ROUND, "technology": name, "delivery_year": str(r["delivery_year"]),
                     "strike_price_gbp_per_mwh_2012": float(r["strike_price_gbp_per_mwh_2012"]),
                     "capacity_mw": 1.0, "price_basis": "2012 prices",
                     "source": "data/market/lccc/cfd_strike_prices_by_round.csv (scripts/market/build_ar8_modelled_strikes.py)",
                     "notes": "the 2030 chain's modelled AR8 strike for the 2029-30 cohort (user decision, 13 Nov 2026)"})
    live = known[pd.to_numeric(known["live_share"], errors="coerce").fillna(0.0).gt(0) & known["scheme"].eq("CfD")]
    for _, r in live.iterrows():
        name = TECHNOLOGY_TABLE_NAME.get(str(r["technology"]))
        if name is None:
            raise ContractBookError(f"known contract {r['contract_id']} has no strike-table technology")
        rows.append({"cfd_round": str(r["cfd_round"]), "technology": name,
                     "delivery_year": str(r.get("start_year", "")),
                     "strike_price_gbp_per_mwh_2012": float(r["strike_gbp_per_mwh_2012"]),
                     "capacity_mw": float(r["live_mw"]), "price_basis": "2012 prices",
                     "source": str(r["source"]),
                     "notes": f"{r['contract_id']} phase {int(r['phase'])} ({r['technology']}, {r['region']}), "
                              f"{float(r['live_mw']):.1f} live MW in the model year"})
    frame = pd.DataFrame(rows)
    if frame.duplicated(["cfd_round", "technology", "delivery_year", "notes"]).any():
        raise ContractBookError("the 2050 strike table repeats an entry")
    return frame


# ── the book ─────────────────────────────────────────────────────────────────

def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


#: The CfD contract register: each contract's own technology (11 Nov 2026).
CFD_CONTRACTS = Path(__file__).resolve().parents[2] / "data/market/support/cfd_contracts.csv"


@lru_cache(maxsize=1)
def _contract_technology() -> dict:
    """contract_id -> its strike-table technology through its own carrier (``cfd_strike_table.carrier_to_technology``).

    A contract spread over register rows of another carrier settles at its own technology (2040, 11 Nov 2026: the
    AR3 advanced-conversion and AR4 energy-from-waste contracts sit on FES "other renewables" rows, carrier biogas,
    and are still live; by 2050 they have ended).
    """
    from scripts.market.cfd_strike_table import carrier_to_technology

    if not CFD_CONTRACTS.exists():
        return {}
    book = pd.read_csv(CFD_CONTRACTS, comment="#", usecols=["contract_id", "carrier"])
    return {str(c): carrier_to_technology(k) for c, k in zip(book["contract_id"], book["carrier"])}


def build_contract_book(assets: pd.DataFrame, table, *, model_year: int, snapshot: Mapping | None = None,
                        params: Mapping | None = None, vintage_table: pd.DataFrame | None = None,
                        insurance_config: Mapping | None = None, learning: bool = False,
                        rate_path: str = "flat", basis: Mapping | None = None,
                        calibration: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per register row holding a live CfD or DPA in the model year.

    Args:
        assets: The vintaged register's asset inputs (``build_asset_finance_register``).
        table: The 2050 ``StrikeTable`` (:func:`strike_table_frame` through ``load_strike_table``).
        model_year: The register's model year.
        snapshot: The workbook snapshot (DPA private rates).
        params: ``support_regime_2050.json``.
        vintage_table: The vintage cost table.
        insurance_config: The finance layer's D4(a) block (the study config's).
        learning: Price DPA capex from the learning column.
        rate_path: ``flat`` or ``neso_wind_ramp`` (DPA rates; no effect on thermal rows).
        basis: Decision 41's capital basis (``vintage_capital_2050.basis_from_config``); a term
            basis reprices each DPA's availability rate (:func:`basis_dpa_apr`). None, or the
            economic-life basis, keeps ``support_regime_2050.dpa_contract``'s rate.
        calibration: The break-even calibration (``vintage_capital_2050.load_calibration``):
            a calibrated DPA cohort's rate replaces its basis rate.

    Returns:
        The book (:data:`BOOK_COLUMNS`).

    Raises:
        ContractBookError: A CfD unit whose round has no entry (a ``future`` fallback is
            refused: an unlabelled 2050 strike is an error, not a guess), a DPA on a carrier
            the scheme does not cover, or cohort shares that double count an asset.
    """
    from scripts.public_ownership import support_regime_2050 as sr
    from scripts.public_ownership.settlement import vintaged_settlement_share

    if missing := set(REGISTER_COLUMNS) - set(assets.columns):
        raise ContractBookError(f"the register lacks columns {sorted(missing)}; it is not a vintaged register")
    support = _text(assets["support_type"])
    live = pd.to_numeric(assets["cfd_live_share"], errors="coerce").fillna(0.0)
    held = support.isin(BOOK_SCHEMES) & _text(assets["cfd_round"]).ne("") & live.gt(0.0)
    held &= assets["component_type"].astype(str).isin(("Generator", "Link"))
    share = vintaged_settlement_share(assets)
    rows = []
    params = sr.load_parameters() if params is None else params
    if snapshot is None and support.eq(SCHEME_DPA).any():
        from scripts.market.cfd_strikes_2050 import load_snapshot

        snapshot = load_snapshot()
    contracts: dict = {}
    term_basis = bool(basis and basis.get("enabled"))
    if term_basis:
        from scripts.public_ownership import vintage_capital_2050 as vk

        capture = vk.load_capture_prices(basis["capture_prices"]) if basis.get("capture_prices") else None
        dpa_calibrated = vk.calibrated_dpa_rates(calibration, amortisation=basis["amortisation"])
    for index in assets.index[held]:
        a = assets.loc[index]
        scheme = str(support[index])
        unit = str(a["asset_id"])
        row = {column: np.nan for column in BOOK_COLUMNS}
        row.update(unit=unit, network_asset=unit.split("::", 1)[0], component_type=str(a["component_type"]),
                   carrier=str(a["carrier"]), stock_flow_technology=str(a["stock_flow_technology"]),
                   build_year=int(float(a["build_year"])), capacity_mw=float(a["capacity_mw"]), scheme=scheme,
                   cfd_round=str(a["cfd_round"]),
                   cfd_contract_id=str(a.get("cfd_contract_id")) if pd.notna(a.get("cfd_contract_id")) else "",
                   settlement_share=float(share[index]), live_share=float(live[index]))
        if not np.isfinite(row["settlement_share"]) or row["settlement_share"] <= 0:
            raise ContractBookError(f"{unit}: no settlement share (cohort_share x site_cohort_share)")
        if scheme == SCHEME_CFD:
            if row["component_type"] != "Generator":
                raise ContractBookError(f"{unit}: a CfD on a {row['component_type']}; CfDs settle generators")
            contract = row["cfd_contract_id"].split("#", 1)[0]
            key, fallback = table.resolve(row["carrier"], row["cfd_round"], _contract_technology().get(contract))
            if key is None or fallback:
                raise ContractBookError(f"{unit}: no 2050 strike for ({row['carrier']}, {row['cfd_round']})")
            row.update(strike_key=f"{key[0]}_{key[1]}", strike_gbp_per_mwh_2024=float(table.strikes[key]),
                       energy_factor=row["settlement_share"] * row["live_share"],
                       source="cfd_strikes_2050 / support_regime_2050 strike via the 2050 strike table")
        else:
            dpa_carrier = DPA_CARRIERS.get(row["carrier"])
            if dpa_carrier is None:
                raise ContractBookError(f"{unit}: a DPA on carrier {row['carrier']!r}; DPAs cover {sorted(DPA_CARRIERS)}")
            technology = row["stock_flow_technology"] if dpa_carrier == "Hydrogen" else None
            cache_key = (dpa_carrier, technology, row["build_year"])
            if cache_key not in contracts:
                contracts[cache_key] = sr.dpa_contract(
                    dpa_carrier, row["build_year"], 1.0, snapshot=snapshot, params=params, model_year=int(model_year),
                    vintage_table=vintage_table, insurance_config=insurance_config, learning=learning,
                    rate_path=rate_path, technology=technology)
                if term_basis:
                    from dataclasses import replace

                    c = contracts[cache_key]
                    apr = basis_dpa_apr(dpa_carrier, technology, row["build_year"], basis=basis, snapshot=snapshot,
                                        insurance_config=insurance_config, capture=capture, params=params,
                                        learning=learning, vintage_table=vintage_table)
                    source = f"{c.source} | decision 41 basis {basis['amortisation']}"
                    key3 = (dpa_carrier, technology or "", row["build_year"])
                    if key3 in dpa_calibrated:
                        base, value = dpa_calibrated[key3]
                        vk.check_base(base, apr, f"DPA {key3}")
                        apr, source = value, f"{source}; calibrated to break even (decision 41 (b)) from {base:.4f}"
                    contracts[cache_key] = replace(c, apr_gbp_per_kw_year=apr,
                                                   fixed_cost_gbp_per_kw_year=apr * c.expected_availability,
                                                   source=source)
            c = contracts[cache_key]
            row.update(dpa_carrier=dpa_carrier, dpa_technology=technology or "",
                       dpa_apr_gbp_per_kw_year=c.apr_gbp_per_kw_year,
                       dpa_expected_availability=c.expected_availability, dpa_term_years=c.term_years,
                       dpa_margin_gbp_per_mwh=c.margin_gbp_per_mwh, dpa_gainshare_share=c.gainshare_share,
                       dpa_gainshare_threshold_gbp_per_kw_year=c.gainshare_threshold_gbp_per_kw_year,
                       dpa_private_rate=c.private_rate, dpa_fixed_cost_gbp_per_kw_year=c.fixed_cost_gbp_per_kw_year,
                       energy_factor=row["settlement_share"], source=c.source)
        rows.append(row)
    book = pd.DataFrame(rows, columns=list(BOOK_COLUMNS))
    if book["unit"].duplicated().any():
        raise ContractBookError("the register repeats an asset id")
    return book


def read_contract_book(path) -> pd.DataFrame:
    """The book file, checked (cached on path and modification time)."""
    path = Path(path)
    return _read_book(str(path), path.stat().st_mtime_ns).copy()


@lru_cache(maxsize=4)
def _read_book(path: str, _mtime: int) -> pd.DataFrame:
    book = pd.read_csv(path, low_memory=False)
    if missing := set(BOOK_COLUMNS) - set(book.columns):
        raise ContractBookError(f"{path} lacks book columns {sorted(missing)}")
    for column in ("unit", "network_asset", "component_type", "carrier", "scheme", "cfd_round"):
        book[column] = _text(book[column])
    if not book["scheme"].isin(BOOK_SCHEMES).all():
        raise ContractBookError(f"{path}: schemes other than {BOOK_SCHEMES}")
    for column in ("settlement_share", "live_share", "energy_factor"):
        values = pd.to_numeric(book[column], errors="coerce")
        if not (values.gt(0) & values.le(1.0 + 1e-9)).all():
            raise ContractBookError(f"{path}: {column} must lie in (0, 1]")
    return book


def contract_book_from_rule(rule: Mapping | None) -> pd.DataFrame | None:
    """The book a bill rule names (``model_linked_rules.cfd.contract_book``), or None (every 2030 bill)."""
    path = (rule or {}).get("contract_book")
    return read_contract_book(ROOT / path if not Path(path).is_absolute() else path) if path else None


# ── CfD units ────────────────────────────────────────────────────────────────

def cfd_units(book: pd.DataFrame, generators: pd.DataFrame, dispatch: pd.DataFrame | None = None,
              *, require_all: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The book's CfD units as network generators: the frame ``compute_cfd_payments`` reads, and their dispatch.

    Args:
        book: :func:`read_contract_book` output.
        generators: Network generators (index the generator name; ``carrier`` and ``bus``).
        dispatch: Network dispatch (snapshots x generators, MW); None returns the frame only.
        require_all: Refuse a book unit whose network asset the network does not carry.

    Returns:
        ``(units, unit_dispatch)``: ``units`` is indexed by unit with ``support_type`` (CfD),
        ``carrier``, ``bus``, ``cfd_round``, ``cfd_support_start``, ``network_asset`` and
        ``energy_factor``; ``unit_dispatch`` is the network asset's dispatch x ``energy_factor``.
    """
    cfd = book.loc[book["scheme"].eq(SCHEME_CFD) & book["component_type"].eq("Generator")]
    present = cfd["network_asset"].isin(generators.index)
    if require_all and not present.all():
        missing = cfd.loc[~present, "network_asset"].drop_duplicates().tolist()
        raise ContractBookError(f"{len(missing)} contract-book assets are not in the network (e.g. {missing[:3]}); "
                                "the book and the network disagree")
    cfd = cfd.loc[present]
    carrier = generators["carrier"].reindex(cfd["network_asset"]).fillna("").astype(str).to_numpy()
    if not np.array_equal(carrier, cfd["carrier"].to_numpy()):
        raise ContractBookError("contract-book carriers differ from the network's")
    units = pd.DataFrame({
        "support_type": SCHEME_CFD, "carrier": cfd["carrier"].to_numpy(),
        "bus": (generators["bus"].reindex(cfd["network_asset"]).fillna("").astype(str).to_numpy()
                if "bus" in generators.columns else ""),
        "cfd_round": cfd["cfd_round"].to_numpy(), "cfd_support_start": "",
        "network_asset": cfd["network_asset"].to_numpy(),
        "energy_factor": pd.to_numeric(cfd["energy_factor"], errors="coerce").to_numpy(dtype=float),
    }, index=pd.Index(cfd["unit"].to_numpy(), name=None))
    # The contract's own strike-table technology, as the book resolved it (11 Nov 2026): compute_cfd_payments reads it.
    from scripts.market.cfd_strike_table import CFD_TECHNOLOGY_COLUMN

    ids = (cfd["cfd_contract_id"].fillna("").astype(str).str.split("#", n=1).str[0]
           if "cfd_contract_id" in cfd.columns else pd.Series("", index=cfd.index))
    units[CFD_TECHNOLOGY_COLUMN] = ids.map(_contract_technology()).fillna("").to_numpy()
    if dispatch is None:
        return units, None
    base = dispatch.reindex(columns=units["network_asset"].to_numpy(), fill_value=0.0)
    values = base.to_numpy(dtype=float) * units["energy_factor"].to_numpy()[None, :]
    return units, pd.DataFrame(values, index=dispatch.index, columns=units.index)


# ── DPAs ─────────────────────────────────────────────────────────────────────

ENERGY_TERMS = ("output_mwh", "variable_top_up_gbp", "market_revenue_gbp", "short_run_cost_gbp")


def dpa_energy_terms(dispatch_mwh: pd.DataFrame, price: pd.Series, plant_srmc: pd.DataFrame,
                     reference_srmc, margin_gbp_per_mwh: float) -> pd.DataFrame:
    """Per network asset, the year's dispatch-dependent DPA sums (additive over months).

    ``top_up_h = max(SRMC_plant,h - SRMC_reference,h + margin, 0)`` (the variable payment
    keeping the plant just ahead of the reference CCGT in merit, decision 22), and the
    market revenue and short-run cost the gainshare is measured on.

    Args:
        dispatch_mwh: Weighted MWh per snapshot and asset (generation, clipped at zero by the caller).
        price: The Stage 1 price the plant sells at, by snapshot.
        plant_srmc: GBP/MWh per snapshot and asset (fuel, carbon and VOM as the model dispatches it).
        reference_srmc: The reference unabated CCGT's GBP/MWh (scalar or by snapshot).
        margin_gbp_per_mwh: The declared margin.

    Returns:
        One row per asset with :data:`ENERGY_TERMS`.
    """
    out = dispatch_mwh.to_numpy(dtype=float)
    srmc = plant_srmc.reindex(index=dispatch_mwh.index, columns=dispatch_mwh.columns).to_numpy(dtype=float)
    if not np.isfinite(srmc).all():
        raise ContractBookError("DPA plant short-run costs are missing for some assets or hours")
    reference = np.broadcast_to(np.asarray(reference_srmc, dtype=float).reshape(-1, 1)
                                if np.ndim(reference_srmc) else float(reference_srmc), out.shape)
    top_up = np.maximum(srmc - reference + float(margin_gbp_per_mwh), 0.0)
    p = price.reindex(dispatch_mwh.index).to_numpy(dtype=float)[:, None]
    return pd.DataFrame({"output_mwh": out.sum(axis=0), "variable_top_up_gbp": (top_up * out).sum(axis=0),
                         "market_revenue_gbp": (p * out).sum(axis=0), "short_run_cost_gbp": (srmc * out).sum(axis=0)},
                        index=dispatch_mwh.columns)


def settle_dpa_book(book: pd.DataFrame, terms: pd.DataFrame, *, period_share: float = 1.0) -> pd.DataFrame:
    """Each DPA unit's payments on its asset's annual energy terms (``support_regime_2050.dpa_payments``).

    A unit takes ``energy_factor`` (its settlement share) of its asset's terms and its own
    contract's availability rate and capacity; ``dpa_payments`` applies the live share.

    Args:
        book: The book (DPA rows are used).
        terms: :func:`dpa_energy_terms` summed over the year, indexed by network asset.
        period_share: The settled hours over the calendar year's (1 for a full year).

    Returns:
        Per unit: the ``dpa_payments`` fields plus ``unit``.
    """
    from scripts.public_ownership.support_regime_2050 import DPAContract, dpa_payments

    dpa = book.loc[book["scheme"].eq(SCHEME_DPA)]
    rows = []
    for _, u in dpa.iterrows():
        asset = terms.loc[u["network_asset"]] if u["network_asset"] in terms.index else pd.Series(0.0, index=ENERGY_TERMS)
        k = float(u["energy_factor"])
        contract = DPAContract(
            carrier=str(u["dpa_carrier"]), build_year=int(u["build_year"]), capacity_mw=float(u["capacity_mw"]),
            apr_gbp_per_kw_year=float(u["dpa_apr_gbp_per_kw_year"]),
            expected_availability=float(u["dpa_expected_availability"]), term_years=int(u["dpa_term_years"]),
            live_share=float(u["live_share"]), efficiency=np.nan, variable_om_gbp_per_mwh=np.nan,
            residual_emission_tco2_per_mwh=np.nan, margin_gbp_per_mwh=float(u["dpa_margin_gbp_per_mwh"]),
            gainshare_share=float(u["dpa_gainshare_share"]),
            gainshare_threshold_gbp_per_kw_year=float(u["dpa_gainshare_threshold_gbp_per_kw_year"]),
            private_rate=float(u["dpa_private_rate"]), fixed_cost_gbp_per_kw_year=float(u["dpa_fixed_cost_gbp_per_kw_year"]),
            source=str(u["source"]))
        result = dpa_payments(contract, output_mwh=k * float(asset["output_mwh"]),
                              variable_top_up_gbp=k * float(asset["variable_top_up_gbp"]),
                              market_revenue_gbp=k * float(asset["market_revenue_gbp"]),
                              short_run_cost_gbp=k * float(asset["short_run_cost_gbp"]), period_share=period_share)
        rows.append({"unit": u["unit"], **result})
    return pd.DataFrame(rows)


def write_support_summary(path, settlement: pd.DataFrame, dpa_units: pd.DataFrame, *, treatment: str,
                          book_path=None, period_share: float = 1.0) -> dict:
    """The settlement's support totals by scheme, which the 2050 bill's ``dpa`` line reads.

    Args:
        path: Output JSON.
        settlement: The register-row settlement.
        dpa_units: :func:`settle_dpa_book` output.
        treatment: How the CfD curtailment compensation was counted.
        book_path: The book settled.
        period_share: As :func:`settle_dpa_book`.

    Returns:
        The summary.
    """
    support = _text(settlement["support_type"])
    cfd = settlement.loc[support.eq(SCHEME_CFD)]
    parts = {k: float(dpa_units[k].sum()) if len(dpa_units) else 0.0
             for k in ("availability_payment_gbp", "variable_payment_gbp", "gainshare_gbp", "net_support_cost_gbp")}
    by_carrier = ({c: float(v) for c, v in dpa_units.groupby("carrier")["net_support_cost_gbp"].sum().items()}
                  if len(dpa_units) else {})
    summary = {
        "schema": "SETTLE2050 support settlement summary (contract_book_2050.write_support_summary)",
        "price_year": 2024, "contract_book": str(book_path) if book_path else None, "period_share": period_share,
        "cfd": {"support_gbp": float(cfd["cfd_support_gbp"].sum()),
                "curtailment_compensation_gbp": float(cfd["cfd_curtailment_compensation_gbp"].sum()),
                "compensation_treatment": treatment, "units": int(len(cfd)),
                "by_carrier_support_gbp": {str(c): float(v) for c, v in
                                           cfd.groupby("carrier")["cfd_support_gbp"].sum().items()}},
        "dpa": {**parts, "by_carrier_net_support_gbp": by_carrier, "units": int(len(dpa_units)),
                "dpa_support_in_settlement_gbp": float(settlement.get("dpa_support_gbp", pd.Series(0.0)).sum())},
    }
    if abs(summary["dpa"]["dpa_support_in_settlement_gbp"] - parts["net_support_cost_gbp"]) > TOLERANCE_GBP:
        raise ContractBookError("the settlement's dpa_support_gbp does not equal the DPA units' net support")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_dpa_line(rule: Mapping | None) -> dict | None:
    """The bill's DPA line input (``model_linked_rules.dpa``), or None when the bill carries none (2030)."""
    if not rule:
        return None
    path = rule.get("settlement_summary")
    if not path:
        raise ContractBookError("model_linked_rules.dpa needs settlement_summary (the 2050 support settlement)")
    full = ROOT / path if not Path(path).is_absolute() else Path(path)
    summary = json.loads(full.read_text(encoding="utf-8"))
    return {"net_support_gbp": float(summary["dpa"]["net_support_cost_gbp"]), "summary": summary, "path": str(path)}


# ── command line ─────────────────────────────────────────────────────────────

def _insurance_from(config_path):
    from scripts.market.cfd_strikes_2050 import finance_insurance_config

    return finance_insurance_config(config_path) if config_path else finance_insurance_config()


def _variant_frame(variant: str, *, snapshot, insurance, model_year: int = 2050) -> pd.DataFrame:
    """The strike table of one Y11 variant (decision 6's economic-life basis and its sensitivities)."""
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import support_regime_2050 as sr

    variants = {"central": {}, "neso_wind_ramp": {"rate_path": "neso_wind_ramp"}, "learning": {"learning": True},
                "contract_term": {"amortisation": "contract_term"}}
    if variant not in variants:
        raise ContractBookError(f"strike variant {variant!r}; one of {sorted(variants)}")
    kwargs = dict(variants[variant], insurance_config=insurance)
    strikes = cfd50.strike_table_2050(snapshot=snapshot, variants={variant: kwargs})
    params = sr.load_parameters()
    beccs_kwargs = {k: v for k, v in kwargs.items() if k in ("rate_path", "learning", "amortisation")}
    beccs = pd.DataFrame([sr.beccs_strike(y, snapshot=snapshot, params=params, insurance_config=insurance,
                                          **beccs_kwargs)
                          for y in range(cfd50.FIRST_MODELLED_BUILD_YEAR, model_year + 1)])
    return strike_table_frame(strikes, cfd50.known_contracts_2050(model_year), variant=variant, beccs=beccs)


def basis_strike_rows(basis: Mapping, *, snapshot, insurance_config, model_year: int = 2050,
                      rate_path: str = "flat") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-AR7 strikes and BECCS strikes on a decision-41 basis (``vintage_capital_2050``), before calibration.

    ``rate_path`` (FES 2050 decision 54): ``neso_wind_ramp`` prices the wind vintages at NESO's
    FES 2025 step-down (``cfd_strikes_2050.levelised_strike``); thermal and BECCS rows are unchanged.

    Returns:
        ``(strikes, beccs)`` in the shapes :func:`strike_table_frame` reads (variant ``central``).
    """
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import support_regime_2050 as sr
    from scripts.public_ownership import vintage_capital_2050 as vk

    mode = basis["amortisation"]
    capture = vk.load_capture_prices(basis["capture_prices"]) if basis.get("capture_prices") else None
    vintage, inputs, hurdles = cfd50.vc.load_vintage_costs(), cfd50.load_strike_inputs(), cfd50.vc.load_hurdle_rates()
    rows = []
    for technology in cfd50.MAIN_TECHNOLOGIES:
        if technology not in PAR7_TECHNOLOGIES:
            continue
        for year in range(cfd50.FIRST_MODELLED_BUILD_YEAR, model_year + 1):
            row = vk.strike(technology, year, amortisation=mode, snapshot=snapshot, capture=capture,
                            insurance_config=insurance_config, vintage_table=vintage, strike_inputs=inputs,
                            hurdle_table=hurdles, rate_path=rate_path)
            row["variant"] = "central"
            rows.append(row)
    params = sr.load_parameters()
    beccs = []
    for year in range(cfd50.FIRST_MODELLED_BUILD_YEAR, model_year + 1):
        row = sr.beccs_strike(year, snapshot=snapshot, params=params, insurance_config=insurance_config,
                              amortisation="economic_life" if mode == vk.ECONOMIC_LIFE else "contract_term")
        beccs.append(vk.with_tail(row, row["technology"], year, amortisation=mode, capture=capture,
                                  insurance_config=insurance_config, vintage_table=vintage, strike_inputs=inputs))
    return pd.DataFrame(rows), pd.DataFrame(beccs)


def basis_strike_frame(basis: Mapping, *, snapshot, insurance_config, model_year: int = 2050,
                       calibration: pd.DataFrame | None = None, rate_path: str = "flat") -> pd.DataFrame:
    """The 2050 contract strike table on a decision-41 basis, with the break-even calibration applied when given.

    A calibrated ``PAR7_<year>`` entry replaces its basis strike; the calibration's base is
    checked against the basis (a stale file fails). Known contracts are unchanged.
    ``rate_path`` (decision 54): the private rate path the basis strikes are priced at; its
    calibration is the one derived on that path's private case. Decision 71: the calibration's auction
    margin must be the basis's ``auction_margin_share`` (0 without the key).
    """
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import vintage_capital_2050 as vk

    strikes, beccs = basis_strike_rows(basis, snapshot=snapshot, insurance_config=insurance_config, model_year=model_year,
                                       rate_path=rate_path)
    frame = strike_table_frame(strikes, cfd50.known_contracts_2050(model_year), variant="central", beccs=beccs)
    modelled_rows = frame["cfd_round"].astype(str).str.startswith(cfd50.MODELLED_ROUND_PREFIX) & frame["technology"].ne("beccs")
    # Stale method text (decision 61): these strikes are decision 41's basis, not decision 6's levelising.
    frame.loc[modelled_rows, "source"] = (f"scripts/public_ownership/vintage_capital_2050.py strike on the decision-41 "
                                          f"basis ({basis['amortisation']}, rate path {rate_path})")
    frame["basis_strike_gbp_per_mwh_2024"] = frame["strike_price_gbp_per_mwh_2012"] * cfd50.CPI_2012_TO_2024
    # Decision 71: the calibration must carry the basis's auction margin (a stale file is refused).
    vk.check_calibration_margin(calibration, basis.get(vk.AUCTION_MARGIN_KEY, 0.0))
    margin = vk.calibration_margin(calibration)
    margin_note = f"; decision 71: auction margin {margin:g} of contract revenue" if margin > 0 else ""
    calibrated = vk.calibrated_strikes(calibration, amortisation=basis["amortisation"])
    if calibrated:
        modelled = frame["cfd_round"].astype(str).str.startswith(cfd50.MODELLED_ROUND_PREFIX)
        for i in frame.index[modelled]:
            key = (str(frame.at[i, "cfd_round"]), str(frame.at[i, "technology"]))
            if key not in calibrated:
                continue
            base, value = calibrated[key]
            vk.check_base(base, float(frame.at[i, "basis_strike_gbp_per_mwh_2024"]), f"{key[0]}_{key[1]}")
            frame.at[i, "strike_price_gbp_per_mwh_2012"] = value / cfd50.CPI_2012_TO_2024
            frame.at[i, "notes"] = (f"{frame.at[i, 'notes']}; decision 41 (b): calibrated to break even on the "
                                    f"modelled output, {base:.4f} -> {value:.4f} GBP/MWh 2024{margin_note}")
    frame["notes"] = frame["notes"].astype(str) + np.where(
        frame["cfd_round"].astype(str).str.startswith(cfd50.MODELLED_ROUND_PREFIX),
        f"; basis {basis['amortisation']} (decision 41)", "")
    return frame.drop(columns=["basis_strike_gbp_per_mwh_2024"])


def basis_dpa_apr(dpa_carrier: str, technology: str | None, build_year: int, *, basis: Mapping, snapshot,
                  insurance_config, capture: pd.DataFrame | None = None, params=None, learning: bool = False,
                  vintage_table=None) -> float:
    """A DPA vintage's availability rate (GBP/kW/yr) on a decision-41 basis, before calibration.

    The scheme's carrier mix (``HYDROGEN_TECHNOLOGY_SPLIT``) when ``technology`` is None, as
    ``support_regime_2050.dpa_contract`` prices it.
    """
    from scripts.public_ownership import support_regime_2050 as sr
    from scripts.public_ownership import vintage_capital_2050 as vk

    params = sr.load_parameters() if params is None else params
    dpa = params["dpa"]
    split = (sr.HYDROGEN_TECHNOLOGY_SPLIT if dpa_carrier == "Hydrogen" else {dpa["technologies"][dpa_carrier]: 1.0})
    if technology:
        split = {technology: 1.0}
    fixed = sum(share * vk.dpa_annual_fixed(t, build_year, amortisation=basis["amortisation"], snapshot=snapshot,
                                            capture=capture, insurance_config=insurance_config, learning=learning,
                                            vintage_table=vintage_table, params=params)["annual_fixed_gbp_per_kw_year"]
                for t, share in split.items())
    return fixed / sr._value(dpa["expected_availability"])


def rate_path_strike_offsets(flat_table, variant_table, technologies=None) -> dict[tuple[str, str], float]:
    """FES 2050 decision 54: per post-AR7 strike entry, the flat table's strike less the rate-path table's (GBP/MWh 2024).

    The cut a case on that private rate path takes on every contract of the entry, whoever owns it
    (the contract is struck on that path's private cost of capital). Only entries both tables carry
    and that differ are returned. ``technologies`` (the path's ``RATE_PATH_TECHNOLOGIES``): an entry
    of any other technology must not differ (review 2, 3 Oct 2026: the path moves only its own
    technologies' cost of capital), and is refused if it does.
    """
    from scripts.market import cfd_strikes_2050 as cfd50

    out = {}
    for key, strike in flat_table.strikes.items():
        if not str(key[0]).startswith(cfd50.MODELLED_ROUND_PREFIX) or key not in variant_table.strikes:
            continue
        diff = float(strike) - float(variant_table.strikes[key])
        if abs(diff) > 1e-9:
            if technologies is not None and str(key[1]) not in set(technologies):
                raise ContractBookError(f"rate-path offset {diff:+.4f} GBP/MWh on {key}, a technology the path does "
                                        f"not move ({sorted(technologies)}): its calibration must carry the flat one")
            out[(str(key[0]), str(key[1]))] = diff
    return out


def overlay_rate_path_calibration(flat: pd.DataFrame | None, path: pd.DataFrame | None,
                                  technologies) -> pd.DataFrame | None:
    """Review 2 (3 Oct 2026): a rate path's calibration over the flat one, outside the path's technologies.

    A path's calibration is derived for its own technologies only (``RATE_PATH_TECHNOLOGIES``: the
    NESO wind ramp moves wind). Every other cohort keeps the flat calibration, which its basis
    strike shares (the path does not move its cost of capital), so its offset is zero; before, it
    fell back to its uncalibrated basis and took the whole flat calibration delta as an offset.
    """
    if path is None:
        return flat
    if flat is None:
        return path
    keep = set(technologies or ())
    own = path.loc[path["technology"].fillna("").astype(str).isin(keep)]
    rest = flat.loc[~flat["technology"].fillna("").astype(str).isin(keep) & ~flat["key"].isin(own["key"])]
    parts = [part for part in (own, rest) if not part.empty]
    return pd.concat(parts, ignore_index=True) if parts else own


#: Columns of the rate-path offsets file (decision 54).
OFFSET_COLUMNS = ("rate_path", "cfd_round", "technology", "strike_flat_gbp_per_mwh_2024",
                  "strike_rate_path_gbp_per_mwh_2024", "offset_gbp_per_mwh_2024", "calibrated")


def write_rate_path_offsets(config_path, out, *, snapshot_path=None, model_year: int = 2050) -> pd.DataFrame:
    """Decision 54: per private rate path and post-AR7 entry, the flat contract strike less the path's own.

    Each path in ``vintage_capital_2050.strike_calibration_by_rate_path`` is priced on its own basis
    (:func:`basis_strike_frame` at that rate path) with its own calibration (derived on the
    ``private@private_rate_<path>`` case: ``vintage_capital_2050.py calibrate --rate-path``). The
    register and the strike adjustments re-price every contract of a case on that path by the
    offset. A file with no rows is written when no path is configured; a path whose calibration is
    not derived yet takes no offset (the first pass) and is flagged ``calibrated = False``.
    """
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import vintage_capital_2050 as vk

    basis = vk.basis_from_config(_read_config(config_path)) if config_path else vk.basis_from_config(None)
    rows = []
    paths = basis.get("rate_path_calibrations") or {}
    if paths:
        snapshot = cfd50.load_snapshot(snapshot_path) if snapshot_path else cfd50.load_snapshot()
        insurance = _insurance_from(config_path)
        flat = cfd50.strike_table_from_frame(basis_strike_frame(
            basis, snapshot=snapshot, insurance_config=insurance, model_year=model_year,
            calibration=vk.load_calibration(basis["strike_calibration"])))
        frames = write_rate_path_tables(config_path, {p: None for p in paths}, snapshot_path=snapshot_path,
                                        model_year=model_year)
        for name, frame in frames.items():
            calibrated = Path(paths[name]).exists()
            table = cfd50.strike_table_from_frame(frame)
            technologies = vk.RATE_PATH_TECHNOLOGIES.get(name)
            for key, offset in (rate_path_strike_offsets(flat, table, technologies) if calibrated else {}).items():
                rows.append({"rate_path": name, "cfd_round": key[0], "technology": key[1],
                             "strike_flat_gbp_per_mwh_2024": float(flat.strikes[key]),
                             "strike_rate_path_gbp_per_mwh_2024": float(table.strikes[key]),
                             "offset_gbp_per_mwh_2024": offset, "calibrated": True})
            if not calibrated:
                rows.append({"rate_path": name, "cfd_round": "", "technology": "", "strike_flat_gbp_per_mwh_2024": None,
                             "strike_rate_path_gbp_per_mwh_2024": None, "offset_gbp_per_mwh_2024": 0.0,
                             "calibrated": False})
    frame = pd.DataFrame(rows, columns=list(OFFSET_COLUMNS))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def load_rate_path_offsets(path) -> dict[str, dict[tuple[str, str], float]]:
    """``{rate path: {(cfd_round, technology): offset}}`` from :func:`write_rate_path_offsets` (empty without a file)."""
    if not path or not Path(path).exists():
        return {}
    table = pd.read_csv(path)
    out: dict[str, dict] = {}
    for r in table.loc[table["cfd_round"].fillna("").astype(str).ne("")].itertuples(index=False):
        out.setdefault(str(r.rate_path), {})[(str(r.cfd_round), str(r.technology))] = float(r.offset_gbp_per_mwh_2024)
    return out


def write_rate_path_tables(config_path, outputs: Mapping[str, str | None], *, snapshot_path=None,
                           model_year: int = 2050) -> dict[str, pd.DataFrame]:
    """Decision 54: the contract strike table at each rate path's own calibration (written where a path is given).

    Each path takes ``vintage_capital_2050.strike_calibration_by_rate_path.<path>`` (derived on the
    ``private@private_rate_<path>`` case: ``vintage_capital_2050.py calibrate --rate-path``).
    """
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import vintage_capital_2050 as vk

    snapshot = cfd50.load_snapshot(snapshot_path) if snapshot_path else cfd50.load_snapshot()
    insurance = _insurance_from(config_path)
    basis = vk.basis_from_config(_read_config(config_path))
    frames = {}
    for path_name, out in outputs.items():
        calibration_path = (basis.get("rate_path_calibrations") or {}).get(path_name)
        if calibration_path is None:
            raise ContractBookError(f"vintage_capital_2050.strike_calibration_by_rate_path has no {path_name!r}")
        if Path(calibration_path).exists():
            calibration = overlay_rate_path_calibration(vk.load_calibration(basis["strike_calibration"]),
                                                        vk.load_calibration(calibration_path),
                                                        vk.RATE_PATH_TECHNOLOGIES.get(path_name))
            frame = basis_strike_frame(basis, snapshot=snapshot, insurance_config=insurance, model_year=model_year,
                                       calibration=calibration, rate_path=path_name)
        else:
            # Not derived yet (the first pass): the flat contract table, so the path's cases take no
            # strike offset until `vintage_capital_2050.py calibrate --rate-path` has run.
            print(f"WARNING: {calibration_path} not derived yet; the {path_name} table is the flat one (no offset)")
            frame = basis_strike_frame(basis, snapshot=snapshot, insurance_config=insurance, model_year=model_year,
                                       calibration=vk.load_calibration(basis["strike_calibration"]))
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out, index=False)
        frames[path_name] = frame
    return frames


def build_strike_table_file(out, *, config_path=None, snapshot_path=None, variant: str = "central",
                            model_year: int = 2050, bm_out=None) -> pd.DataFrame:
    """Write the 2050 strike table (:func:`strike_table_frame`) for the bill's ``model_linked_rules.cfd.strike_table``.

    Decision 41: when the study config sets ``vintage_capital_2050`` to a term basis, the
    central table is the contract strikes on that basis (:func:`basis_strike_frame`, the
    break-even calibration applied); ``bm_out`` then receives the table the solved BM bid
    on, named by the bill as ``model_linked_rules.cfd.bm_strike_table``: decision 6's
    economic-life central (``vintage_capital_2050.bm_strike_basis: economic_life``, labels
    ``strike_variant: central``) or, decision 49 (b), the contract table itself
    (``bm_strike_basis: calibrated``, labels ``strike_variant: calibrated``).
    """
    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership import vintage_capital_2050 as vk

    snapshot = cfd50.load_snapshot(snapshot_path) if snapshot_path else cfd50.load_snapshot()
    insurance = _insurance_from(config_path)
    basis = vk.basis_from_config(_read_config(config_path)) if config_path else vk.basis_from_config(None)
    if variant == "central" and basis["enabled"]:
        frame = basis_strike_frame(basis, snapshot=snapshot, insurance_config=insurance, model_year=model_year,
                                   calibration=vk.load_calibration(basis["strike_calibration"]))
    else:
        frame = _variant_frame(variant, snapshot=snapshot, insurance=insurance, model_year=model_year)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    if bm_out is not None:
        if variant == "central" and basis["enabled"] and basis.get("bm_strike_basis") == "calibrated":
            bm = frame
        else:
            bm = _variant_frame("central", snapshot=snapshot, insurance=insurance, model_year=model_year)
        Path(bm_out).parent.mkdir(parents=True, exist_ok=True)
        bm.to_csv(bm_out, index=False)
    return frame


def _read_config(config_path) -> dict:
    import yaml

    from config.config_loader import read_config

    return read_config(config_path) or {}


def build_book_file(out, *, assets_path, assumptions_path, config_path=None, run_group=None, snapshot_path=None,
                    model_price_year: int = 2024) -> pd.DataFrame:
    """Write the contract book of a vintaged register (:func:`build_contract_book`)."""
    import yaml

    from scripts.market import cfd_strikes_2050 as cfd50
    from scripts.public_ownership.build_cfd_strike_adjustments import load_bill_strike_table
    from scripts.public_ownership.model_year import frame_model_year

    assets = pd.read_csv(assets_path, low_memory=False)
    model_year = int(frame_model_year(assets))
    table = load_bill_strike_table(assumptions_path, model_price_year)
    learning = False
    if config_path and run_group:
        block = _read_config(config_path).get("public_ownership") or {}
        group = (block.get("reference_run_groups") or {}).get(run_group)
        if group is None:
            raise ContractBookError(f"{config_path} declares no run group {run_group!r}: pass the study's resolved config")
        learning = bool((group.get("vintage_costs") or {}).get("learning", False))
    snapshot = cfd50.load_snapshot(snapshot_path) if snapshot_path else cfd50.load_snapshot()
    basis, calibration = None, None
    if config_path:
        from scripts.public_ownership import vintage_capital_2050 as vk

        basis = vk.basis_from_config(_read_config(config_path))
        calibration = vk.load_calibration(basis["strike_calibration"]) if basis["enabled"] else None
    book = build_contract_book(assets, table, model_year=model_year, snapshot=snapshot,
                               insurance_config=_insurance_from(config_path), learning=learning,
                               basis=basis, calibration=calibration)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    book.to_csv(out, index=False)
    return book


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("strike-table", help="write the 2050 strike table in the published format")
    s.add_argument("--out", required=True)
    s.add_argument("--config", default=None, help="the study's resolved config (insurance, D4(a))")
    s.add_argument("--snapshot", default=None)
    s.add_argument("--variant", default="central")
    s.add_argument("--bm-out", default=None, help="decision 41: also write the table the solved BM bid on")
    s.add_argument("--rate-path-offsets-out", default=None,
                   help="decision 54: write each private rate path's strike offsets (flat less the path's own)")
    # Year-generic (2040, 11 Nov 2026): the study's modelled year; the post-AR7 cohorts run to it.
    s.add_argument("--model-year", type=int, default=2050)
    b = sub.add_parser("book", help="write a vintaged register's contract book")
    b.add_argument("--assets", required=True, help="{group}_asset_inputs.csv of a vintaged register")
    b.add_argument("--assumptions", required=True, help="the 2050 bill assumptions naming the strike table")
    b.add_argument("--out", required=True)
    b.add_argument("--config", default=None, help="the study's resolved config")
    b.add_argument("--run-group", default=None)
    b.add_argument("--snapshot", default=None)
    b.add_argument("--model-price-year", type=int, default=2024)
    args = parser.parse_args(argv)
    if args.command == "strike-table":
        frame = build_strike_table_file(args.out, config_path=args.config, snapshot_path=args.snapshot,
                                        variant=args.variant, bm_out=args.bm_out, model_year=args.model_year)
        print(f"wrote {args.out}: {len(frame)} rows")
        if args.rate_path_offsets_out:
            offsets = write_rate_path_offsets(args.config, args.rate_path_offsets_out, snapshot_path=args.snapshot,
                                              model_year=args.model_year)
            print(f"wrote {args.rate_path_offsets_out}: {len(offsets)} rows")
    else:
        book = build_book_file(args.out, assets_path=args.assets, assumptions_path=args.assumptions,
                               config_path=args.config, run_group=args.run_group, snapshot_path=args.snapshot,
                               model_price_year=args.model_price_year)
        print(f"wrote {args.out}: {len(book)} contract units "
              f"({int(book['scheme'].eq(SCHEME_CFD).sum())} CfD, {int(book['scheme'].eq(SCHEME_DPA).sum())} DPA)")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    main()
