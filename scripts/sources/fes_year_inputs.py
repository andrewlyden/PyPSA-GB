"""A new long-range FES year's hand-made inputs, written by script (user, 11 Nov 2026).

Three commands, each writing only files that do not exist (``--force`` overwrites):

``seeds``
    The strike calibration (and its auction-margin and wind-ramp variants) and the capture prices the year's first
    balancing pass bids on: 2050's rows with a build year at or before the year, marked SEED. Re-derived after the
    solve (``scripts/public_ownership/post_solve_year.py``).
``inputs``
    The year's ancillary services, energy-from-waste gate fees, network RAV roll-forward and guarantee pass-through,
    from the 2050 files as 2040's were made by hand: every value year-neutral and held (2024 money), the FES context
    re-read for the year (fleet, waste, peak demand), the pass-through share derived from the price controls'
    trailing-average windows.
``performance``
    The year's renewable performance factors: ES1 gross energy (generation + curtailment) over the modelled
    available energy of the year's renewables network, with that network's own factors divided out (so any build of
    the year serves, not only one at factor 1.0). ``--write`` puts the rows in ``data/FES/fes_year_parameters.csv``.

    python -m scripts.sources.fes_year_inputs seeds --year 2045
    python -m scripts.sources.fes_year_inputs inputs --year 2045
    python -m scripts.sources.fes_year_inputs performance --year 2045 [--write]
    python -m scripts.sources.fes_year_inputs parameters --year 2045 [--write]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

FIN = "data/public_ownership/financing"
NET = "data/market/network_cost_sources"
SOURCE_YEAR = 2050
#: The first year a guarantee reprices debt (user decision 6: debt issued or refinanced from 2026).
GUARANTEE_FROM = 2026
#: Ofgem's trailing averages: transmission RAV-weighted over 14 years, distribution (ED2) simple over 17.
TRAILING_YEARS = {"transmission": 14, "distribution": 17}
PERFORMANCE_CARRIERS = ("wind_offshore", "wind_onshore", "solar_pv")


def _targets(year: int) -> dict:
    path = ROOT / f"resources/FES/fes2025_holistic_transition_{int(year)}_targets.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} (rule build_fes2025_energy_targets) is needed for {year}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str, force: bool) -> bool:
    if path.exists() and not force:
        print(f"kept {path.relative_to(ROOT)} (exists; --force overwrites)")
        return False
    path.write_text(text, encoding="utf-8", newline="")
    print(f"wrote {path.relative_to(ROOT)}")
    return True


# ── seeds ────────────────────────────────────────────────────────────────────

def seed_text(year: int) -> str:
    return (f"SEED for the first {year} pass ({date.today().isoformat()}): the {SOURCE_YEAR} file's row, re-derived "
            f"on the {year} register after the {year} solve (scripts/public_ownership/post_solve_year.py)")


def seeds(year: int, force: bool = False) -> None:
    y = int(year)
    for suffix in ("", "_auction_margin_none", "_auction_margin_high", "_neso_wind_ramp"):
        src = ROOT / FIN / f"strike_calibration_{SOURCE_YEAR}{suffix}.csv"
        if not src.exists():
            continue
        table = pd.read_csv(src)
        table = table.loc[pd.to_numeric(table["build_year"], errors="coerce") <= y].copy()
        table["source"] = seed_text(y)
        _write(ROOT / FIN / f"strike_calibration_{y}{suffix}.csv", table.to_csv(index=False, lineterminator="\n"), force)
    capture = pd.read_csv(ROOT / FIN / f"capture_prices_{SOURCE_YEAR}.csv")
    capture["source"] = seed_text(y)
    _write(ROOT / FIN / f"capture_prices_{y}.csv", capture.to_csv(index=False, lineterminator="\n"), force)


# ── inputs ───────────────────────────────────────────────────────────────────

def guarantee_shares(year: int) -> dict:
    """The share of each price control's cost-of-debt allowance repriced by a guarantee from 2026 in ``year``."""
    out = {}
    for control, n in TRAILING_YEARS.items():
        first = int(year) - n + 1
        out[control] = round(max(0, int(year) - max(first, GUARANTEE_FROM) + 1) / n, 4)
    return out


def ancillary(year: int, targets: dict) -> str:
    table = pd.read_csv(ROOT / FIN / f"ancillary_services_{SOURCE_YEAR}.csv")
    table = table.rename(columns={f"spend_{SOURCE_YEAR}_{k}_gbp_m_2024": f"spend_{k}_gbp_m_2024"
                                  for k in ("central", "low", "high")})
    cap = targets.get("capacity_mw") or {}
    wind = (float(cap.get("wind_offshore", 0.0)) + float(cap.get("wind_onshore", 0.0))) / 1e3
    ic = float((targets.get("interconnectors") or {}).get("capacity_mw") or 0.0) / 1e3
    note = (f" [{year}, {date.today().isoformat()}: the {SOURCE_YEAR} study's projection held for {year} "
            f"(scripts/sources/fes_year_inputs.py); every multiplier is a judgement on the same evidence; FES 2025 HT "
            f"{year}: wind {wind:.0f} GW, interconnectors {ic:.1f} GW.]")
    table["projection_basis"] = table["projection_basis"].astype(str) + note
    return table.to_csv(index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def efw(year: int, targets: dict) -> str:
    doc = json.loads((ROOT / FIN / f"efw_gate_fees_{SOURCE_YEAR}.json").read_text(encoding="utf-8"))
    doc["purpose"] = doc["purpose"].replace(f"the {SOURCE_YEAR} public-ownership", f"the {year} public-ownership")
    proj = dict(doc.pop(f"projection_{SOURCE_YEAR}"))
    gen = {r["carrier"]: r for r in targets["generation"]}
    w = gen.get("waste_to_energy") or {}
    mw, gwh = float(w.get("capacity_mw") or 0.0), float(w.get("generation_gwh") or 0.0)
    per_t = float(doc["mwh_exported_per_tonne"]["central"])
    fee = float(proj["central_gbp_per_tonne_2024"])
    mt = gwh * 1e3 / per_t / 1e6
    proj["rule"] = proj["rule"].replace(f"to {SOURCE_YEAR}", f"to {year}")
    proj["fes_context"] = (f"FES 2025 Holistic Transition {year}: waste_to_energy {mw:,.0f} MW, {gwh:,.0f} GWh "
                           f"(resources/FES/fes2025_holistic_transition_{year}_targets.csv, ES1). At {per_t:.3f} MWh/t "
                           f"that is about {mt:.1f} Mt of waste a year; at {fee:.0f} GBP/t about GBP "
                           f"{mt * fee / 1e3:.2f} bn of gate-fee revenue in {year}.")
    out = {}
    for k, v in doc.items():
        if k == "notes":
            out[f"projection_{year}"] = proj
        out[k] = v
    out.setdefault(f"projection_{year}", proj)
    out["derived_from"] = (f"{FIN}/efw_gate_fees_{SOURCE_YEAR}.json ({date.today().isoformat()}, "
                           f"scripts/sources/fes_year_inputs.py): every value year-neutral (2024 money), only the FES "
                           f"context moved to {year}")
    return json.dumps(out, indent=2, ensure_ascii=False) + "\n"


def rav(year: int) -> str:
    from scripts.consumer_bill.fes_bill_figures import read

    text = io.open(ROOT / NET / f"rav_roll_forward_{SOURCE_YEAR}.yaml", encoding="utf-8").read()
    peaks = read(year, ROOT)["peak_gw"]
    peak = float(peaks[max(peaks)] if isinstance(peaks, dict) else peaks)
    old = '    values: {"2030": 62.09, "2050": 120.05}'
    if text.count(old) != 1:
        raise ValueError(f"rav_roll_forward_{SOURCE_YEAR}.yaml: the distribution peak line moved; update fes_year_inputs.rav")
    text = text.replace(old, f'    values: {{"2030": 62.09, "{year}": {peak:.3f}, "2050": 120.05}}')
    header = (f"# Network RAV roll-forward past the determined price controls, for model year {year} "
              f"({date.today().isoformat()}, scripts/sources/fes_year_inputs.py): the {SOURCE_YEAR} file with FES 2025 "
              f"HT's own {year} peak added to the distribution path ({peak:.2f} GW, ED1 row 134).\n")
    return header + text


def guarantee(year: int) -> str:
    table = pd.read_csv(ROOT / NET / f"network_guarantee_pass_through_{SOURCE_YEAR}.csv")
    s = guarantee_shares(year)
    rows = table.set_index("setting")
    t_first, d_first = year - TRAILING_YEARS["transmission"] + 1, year - TRAILING_YEARS["distribution"] + 1
    rows.loc["new_and_refinanced", "basis"] = (
        f"User decision 6 generalised to calendar {year} (scripts/sources/fes_year_inputs.py): the guarantee reaches the "
        f"debt issued or refinanced from {GUARANTEE_FROM} to the model year, the whole allowance in the long run.")
    rows.loc["new_debt_low", ["transmission_share", "distribution_share"]] = [s["transmission"], s["distribution"]]
    rows.loc["new_debt_low", "basis"] = (
        f"Low end for {year}: transmission's {TRAILING_YEARS['transmission']}-year RAV-weighted trailing average "
        f"({t_first}-{year}) has {s['transmission']:.4f} of its years from {GUARANTEE_FROM}; distribution's ED2-style "
        f"simple {TRAILING_YEARS['distribution']}-year average ({d_first}-{year}) has {s['distribution']:.4f} "
        "(equal weights, declared).")
    return rows.reset_index().to_csv(index=False, lineterminator="\n")


def inputs(year: int, force: bool = False) -> None:
    y = int(year)
    t = _targets(y)
    _write(ROOT / FIN / f"ancillary_services_{y}.csv", ancillary(y, t), force)
    _write(ROOT / FIN / f"efw_gate_fees_{y}.json", efw(y, t), force)
    _write(ROOT / NET / f"rav_roll_forward_{y}.yaml", rav(y), force)
    _write(ROOT / NET / f"network_guarantee_pass_through_{y}.csv", guarantee(y), force)


# ── performance factors ──────────────────────────────────────────────────────

def available_gwh(network) -> dict:
    """Modelled available energy (GWh) by carrier: p_nom x p_max_pu over the weighted snapshots."""
    g = network.generators
    w = network.snapshot_weightings
    w = w["generators"] if hasattr(w, "columns") and "generators" in w.columns else w
    pmax = network.generators_t.p_max_pu
    out = {}
    for carrier in PERFORMANCE_CARRIERS:
        names = g.index[g["carrier"].eq(carrier)]
        varying = [n for n in names if n in pmax.columns]
        static = [n for n in names if n not in pmax.columns]
        energy = pmax[varying].mul(g.loc[varying, "p_nom"], axis=1).mul(w, axis=0).to_numpy().sum()
        energy += float((g.loc[static, "p_max_pu"] * g.loc[static, "p_nom"]).sum()) * float(w.sum())
        out[carrier] = energy / 1e3
    return out


def installed_factors(year: int) -> dict:
    table = pd.read_csv(ROOT / "data/FES/fes_year_parameters.csv")
    rows = table.loc[table["year"].eq(int(year)) & table["pathway"].eq("HT")].set_index("key")["value"]
    return {c: float(rows.get(f"performance_factor_{c}", 1.0)) for c in PERFORMANCE_CARRIERS}


def performance(year: int, network_path: str | None = None, write: bool = False) -> dict:
    y = int(year)
    path = Path(network_path or ROOT / f"resources/network/FES25_HT_{y}_network_demand_renewables.pkl")
    network = pd.read_pickle(path) if path.suffix == ".pkl" else __import__("pypsa").Network(str(path))
    built = installed_factors(y)
    avail = available_gwh(network)
    gen = {r["carrier"]: r for r in _targets(y)["generation"]}
    out = {}
    for c in PERFORMANCE_CARRIERS:
        at_one = avail[c] / built[c]
        gross = float(gen[c].get("gross_gwh") or gen[c]["generation_gwh"])
        out[c] = {"factor": round(gross / at_one, 4), "gross_gwh": gross, "available_at_one_gwh": at_one,
                  "built_factor": built[c]}
        print(f"{c}: ES1 gross {gross:,.0f} / available at 1.0 {at_one:,.0f} GWh = {gross / at_one:.4f} "
              f"(network built at {built[c]})")
    if write:
        p = ROOT / "data/FES/fes_year_parameters.csv"
        table = pd.read_csv(p, dtype=str, keep_default_na=False)
        for c, r in out.items():
            key = f"performance_factor_{c}"
            source = (f"Availability calibrated to FES 2025 HT {y} ES1 gross energy (generation + curtailment) on the "
                      f"2013 weather year: ES1 gross / modelled available energy of the FES25_HT_{y} renewables network "
                      f"at factor 1.0 ({date.today().isoformat()}; {r['gross_gwh']:,.0f} / {r['available_at_one_gwh']:,.0f} "
                      "GWh; scripts/sources/fes_year_inputs.py performance)")
            mask = table["year"].eq(str(y)) & table["pathway"].eq("HT") & table["key"].eq(key)
            if mask.any():
                table.loc[mask, ["value", "source"]] = [f"{r['factor']}", source]
            else:
                table.loc[len(table)] = {"year": str(y), "pathway": "HT", "key": key, "value": f"{r['factor']}",
                                         "unit": "factor", "source": source}
        table.to_csv(p, index=False, lineterminator="\n")
        print(f"wrote the {y} performance factors to {p.relative_to(ROOT)}")
    return out


# -- the year's parameters (data/FES/fes_year_parameters.csv) -------------------------------------------------

#: Reforming CCS as the FES studies configure it (hydrogen.fes.gas_reforming_ccs).
REFORMING = {"capture_rate": 0.95, "co2_transport_storage_gbp_per_tco2": 34.5, "variable_om_gbp_per_mwh_h2": 0.0}
#: Decision 40 (2050): the annual pass's hydrogen reserve, scaled to another year by its non-power hydrogen demand.
RESERVE_2050_TWH = 4.0


def parameters(year: int, write: bool = False) -> dict:
    """The year's hydrogen and gas-CCS parameters, on the rules 2040's were derived by (10 Nov 2026)."""
    from scripts.hydrogen.fes_hydrogen import future_cost_prices, hydrogen_quantities, reformer_costs

    y = int(year)
    q = hydrogen_quantities(_targets(y))
    q50 = hydrogen_quantities(_targets(SOURCE_YEAR))
    prices = future_cost_prices(ROOT / f"resources/FES/fes2025_future_costs_{y}.json")
    h2 = reformer_costs(prices["gas"], prices["carbon"], REFORMING, q["reforming_efficiency"])["hydrogen_cost_gbp_per_mwh_h2"]
    reserve = RESERVE_2050_TWH * q["non_power_demand_mwh"] / q50["non_power_demand_mwh"]
    today = date.today().isoformat()
    rows = {
        "hydrogen_supply_price_gbp_per_mwh_h2": (64.0, "GBP/MWh_H2 (2024 prices)",
            f"BEIS Hydrogen Production Costs 2021, ATR + CCUS: Chart 6.1 (p.28) shows the 2040 and 2050 levelised costs "
            f"equal, so {y} takes the same 52 GBP/MWh H2 HHV on wholesale gas (2020 prices) x 1.232 (ONS CPI 2020 to "
            f"2024) = 64 ({today}, scripts/sources/fes_year_inputs.py parameters)"),
        "hydrogen_end_of_window_value_gbp_per_mwh_h2": (round(h2, 2), "GBP/MWh_H2 (2024 prices)",
            f"The CCS reformer's short-run cost in FES 2025 HT {y} (scripts/hydrogen/fes_hydrogen.reformer_costs: FES "
            f"2025 gas {prices['gas']:.2f} and carbon {prices['carbon']:.2f} GBP 2024, reforming efficiency "
            f"{q['reforming_efficiency']:.3f}, 95% capture, 34.5 GBP/t CO2 transport and storage) ({today})"),
        "ccs_gas_efficiency": (0.47, "fraction (net)",
            "DESNZ Electricity Generation Costs 2025 Annex A, gas CCS: 47% net efficiency (Annex A note 9: only capex "
            f"varies with commissioning year) ({today})"),
        "ccs_gas_variable_om_gbp_per_mwh": (18.09, "GBP/MWh (2024 prices)",
            "DESNZ Electricity Generation Costs 2025 Annex A: variable O&M 6 GBP/MWh plus CO2 transport and storage at "
            f"34.5 GBP/t on 0.183 x 0.90 / 0.47 = 0.3504 tCO2 per MWh (12.09) ({today})"),
        "hydrogen_reserve_twh": (round(reserve, 2), "TWh_H2",
            f"4 TWh (decision 40, 2050) x FES 2025 HT {y} non-power hydrogen demand / 2050's "
            f"({q['non_power_demand_mwh'] / 1e6:.2f} / {q50['non_power_demand_mwh'] / 1e6:.2f} TWh, calendar basis, "
            f"scripts/hydrogen/fes_hydrogen.hydrogen_quantities) ({today})"),
    }
    for key, (value, unit, _) in rows.items():
        print(f"{y} {key} = {value} {unit}")
    if write:
        path = ROOT / "data/FES/fes_year_parameters.csv"
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        for key, (value, unit, source) in rows.items():
            mask = table["year"].eq(str(y)) & table["pathway"].eq("HT") & table["key"].eq(key)
            if mask.any():
                print(f"kept {y} {key} (already in the table)")
                continue
            table.loc[len(table)] = {"year": str(y), "pathway": "HT", "key": key, "value": str(value), "unit": unit,
                                     "source": source}
        table.to_csv(path, index=False, lineterminator=chr(10))
        print(f"wrote the {y} parameters to {path.relative_to(ROOT)}")
    return {k: v[0] for k, v in rows.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("seeds", "inputs", "performance", "parameters"))
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="overwrite existing files (seeds, inputs)")
    ap.add_argument("--network", default=None, help="performance: the year's renewables network (.pkl or .nc)")
    ap.add_argument("--write", action="store_true", help="performance, parameters: write the rows to fes_year_parameters.csv")
    a = ap.parse_args(argv)
    if a.year >= SOURCE_YEAR:
        print(f"{a.year}: the {SOURCE_YEAR} files are the source; nothing to derive", file=sys.stderr)
        return 2
    if a.command == "seeds":
        seeds(a.year, a.force)
    elif a.command == "inputs":
        inputs(a.year, a.force)
    elif a.command == "parameters":
        parameters(a.year, a.write)
    else:
        performance(a.year, a.network, a.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
