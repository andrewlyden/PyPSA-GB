#!/usr/bin/env python3
"""FES long-range final-dispatch validation against FES 2025 ES1 (plan Y8, acceptance item 2).

Written for HT50 2050 and run for any FES long-range market year (rule ``validate_fes_dispatch``;
the year and pathway come from the study's scenario config).

The FES 2025 Holistic Transition counterpart of ``validate_CP30_dispatch``. It sums the
twelve balancing months of a 2050 market year and reconciles them with ES1 (calendar 2050)
through the shared, unchanged ``scripts/CP30/energy_validation.py``
(``energy_targets("FES2025", ...)``, ``dataset_energy_tables``, ``build_energy_validation``).
Energy gaps WARN; an unmapped ES1 line or a demand mismatch FAILS, as in 2030.

What the 2050 network needs that the CP30 reader does not do, handled here and only here:

* **Interconnectors** are signed DC links priced on the link (IC1), with ``EU_market``
  balance generators on the foreign buses. Imports and exports are measured on the links,
  per cable, on the GB side (imports delivered, exports given up by GB: the B9 convention
  of ``analyze_market``), not netted per foreign bus.
* **Gas supply to the reformer** is a generator on the natural-gas bus, not electricity:
  every generator whose bus is not AC is left out of the ES1 comparison and reported in the
  hydrogen rows.
* **Domestic batteries** are their own carrier in the model and inside ES1's Battery line,
  so the Battery row compares Battery + Domestic Battery.
* **2050 extras:** the hydrogen system (electrolysis, reformer, turbine burn, store) against
  WS1, DACCS against ES1, the demand components against the FES building blocks, storage
  charging against ES1 consumption, and, for every resource carrier, the split of model
  curtailment into Stage 1 spill and BM turn-down (with the Scottish share).

Every row carries its Stage 1 value (``stage1_gwh``) where one exists, and every WARN carries
an ``explanation``. A WARN with no explanation is written as ``UNEXPLAINED`` and the run
fails after writing its outputs: after a re-solve, a new gap must be explained, not passed.

    python scripts/FES/fes_dispatch_validation.py --study FES25_HT_2050 --modelled-year 2050 \
        --output-csv resources/analysis/FES25_HT_2050_fes_energy_validation.csv
"""


import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# The repository root: the working directory under Snakemake (which runs a copy of this file),
# this file's grandparent when run as `python scripts/...`.
for _root in (Path.cwd(), Path(__file__).resolve().parents[2]):
    if (_root / "scripts").is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.CP30.energy_validation import (  # noqa: E402
    CURTAILMENT_CARRIERS,
    DEFAULT_TOLERANCES,
    FES2025_MEMBER_CARRIERS,
    _row,
    build_energy_validation,
    dataset_energy_tables,
    energy_targets,
    energy_validation_failures,
    sum_energy_tables,
    target_column,
)
from scripts.interconnectors.market_coupling import _piecewise_links  # noqa: E402
from scripts.network_build.branch_attributes import scotland_by_geography  # noqa: E402
from scripts.utilities.calendar_year import hours_in_year  # noqa: E402

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
FES2025_WORKBOOK = "data/FES/Future Energy Scenarios 2025 Data Workbook V005_0.xlsx"
# ES1's Battery line includes domestic batteries; the model carries them separately.
STORAGE_FOLD = {"Domestic Battery": "Battery"}
H2_TURBINE, ELECTROLYSIS, REFORMER = "H2_turbine", "electrolysis", "gas_reforming_ccs"
DACCS_CARRIER = "daccs"
RESOURCE_SPLIT_CARRIERS = ("wind_offshore", "wind_onshore", "solar_pv", "marine", "large_hydro")
# WS1 lines (calendar TWh H2) the hydrogen rows compare with.
# Networked only: ES1 electrolysis demand is the networked feedstock, and the model's 15.66 GW
# is the networked fleet (non-networked electrolysis is off the grid).
WS1_ELECTROLYSIS = ("Networked Electrolysis",)
WS1_REFORMER = "Methane Reformation with CCUS"
WS1_POWER = "Power Generation"


# ─────────────────────────────── reading ────────────────────────────────────

def _static(ds, prefix: str, columns) -> pd.DataFrame:
    index = ds[f"{prefix}_i"].values.astype(str)
    return pd.DataFrame({c: ds[f"{prefix}_{c}"].values for c in columns if f"{prefix}_{c}" in ds}, index=index)


def _series(ds, prefix: str, attr: str) -> pd.DataFrame:
    key = f"{prefix}_t_{attr}"
    if key not in ds:
        return pd.DataFrame()
    # Full read then pandas (a label .sel on a long name list is slow); integer snapshot
    # index, never joined to anything timestamped.
    return pd.DataFrame(ds[key].values, columns=ds[f"{key}_i"].values.astype(str))


def _weights(ds) -> np.ndarray:
    if "snapshots_generators" in ds:
        return ds["snapshots_generators"].values.astype(float)
    return np.ones(ds.sizes.get("snapshots", 0))


def _carrier_sum(frame: pd.DataFrame, carriers: pd.Series, w: np.ndarray) -> pd.Series:
    cols = [c for c in frame.columns if c in carriers.index]
    if not cols:
        return pd.Series(dtype=float)
    return frame[cols].mul(w, axis=0).sum().groupby(carriers.reindex(cols).astype(str).to_numpy()).sum()


def interconnector_energy(p0: pd.DataFrame, links: pd.DataFrame, w: np.ndarray) -> dict[str, Any]:
    """Per-cable GB-side import and export energy (MWh) of the signed IC links.

    ``p0 > 0`` is flow from the foreign bus (bus0) into GB: GB receives ``efficiency x p0``.
    On an export PyPSA books ``p1 = -efficiency x p0``, so GB gives up ``efficiency x |p0|``
    (the B9 convention of ``analyze_market``). ``hourly_net`` is the GB net import per hour.
    """
    signed = [link for link in _piecewise_links(links).values() if link in p0.columns]
    if not signed:
        empty = pd.Series(dtype=float)
        return {"import_mwh": empty, "export_mwh": empty, "net_import_mwh": 0.0, "net_export_mwh": 0.0}
    eff = pd.to_numeric(links.loc[signed, "efficiency"], errors="coerce").fillna(1.0).to_numpy()
    flow = p0[signed].astype(float)
    imported = flow.clip(lower=0.0) * eff
    exported = (-flow).clip(lower=0.0) * eff
    net = (imported - exported).sum(axis=1).to_numpy() * w
    return {"import_mwh": imported.mul(w, axis=0).sum(), "export_mwh": exported.mul(w, axis=0).sum(),
            "net_import_mwh": float(net.clip(min=0).sum()), "net_export_mwh": float((-net).clip(min=0).sum())}


def _hydrogen_links(p0: pd.DataFrame, links: pd.DataFrame, w: np.ndarray) -> dict[str, float]:
    """Electricity and hydrogen through the electrolysers, reformer and hydrogen turbines (MWh)."""
    out: dict[str, float] = {}
    carrier = links["carrier"].astype(str)
    eff = pd.to_numeric(links.get("efficiency", 1.0), errors="coerce").fillna(1.0)
    for name, key_in, key_out in ((ELECTROLYSIS, "electrolysis_elec_mwh", "electrolysis_h2_mwh"),
                                  (REFORMER, "reformer_gas_mwh", "reformer_h2_mwh"),
                                  (H2_TURBINE, "h2_turbine_h2_mwh", "h2_turbine_elec_mwh")):
        cols = [c for c in links.index[carrier.eq(name)] if c in p0.columns]
        flow = p0[cols].clip(lower=0.0) if cols else pd.DataFrame()
        out[key_in] = float(flow.mul(w, axis=0).to_numpy().sum()) if cols else 0.0
        out[key_out] = float((flow * eff.reindex(cols).to_numpy()).mul(w, axis=0).to_numpy().sum()) if cols else 0.0
    return out


def _foreign_ask_hours(links: pd.DataFrame, ask_t: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Sum over hours of each tranche group's mean import ask (GBP/MWh x h), for an annual mean."""
    signed = list(_piecewise_links(links).values())
    if not signed:
        return pd.Series(dtype=float)
    static = pd.to_numeric(links.loc[signed, "marginal_cost"], errors="coerce")
    ask = pd.DataFrame(np.tile(static.to_numpy(), (len(w), 1)), columns=signed)
    own = [c for c in signed if c in ask_t.columns]
    if own:
        ask[own] = ask_t[own].to_numpy()
    group = (links.loc[signed, "market_tranche_group"].astype(str) if "market_tranche_group" in links
             else pd.Series(signed, index=signed))
    return ask.mul(w, axis=0).sum().groupby(group.to_numpy()).mean()


def read_balancing_extras(path: str | Path) -> dict[str, Any]:
    """What the shared reader does not take from one balancing month (see the module note)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        w = _weights(ds)
        buses = _static(ds, "buses", ["carrier", "x", "y"])
        gens = _static(ds, "generators", ["carrier", "bus", "p_nom", "p_min_pu"])
        gens["carrier"] = gens["carrier"].astype(str)
        if "p_min_pu" not in gens:
            gens["p_min_pu"] = 0.0
        floor_pu = pd.Series(pd.to_numeric(gens["p_min_pu"], errors="coerce").fillna(0.0).clip(lower=0.0) * w.sum(),
                             index=gens.index)
        p_min_t = _series(ds, "generators", "p_min_pu")
        tv = [c for c in p_min_t.columns if c in gens.index]
        if tv:
            floor_pu.loc[tv] = p_min_t[tv].clip(lower=0.0).mul(w, axis=0).sum()
        must_run = (floor_pu * gens["p_nom"].astype(float)).groupby(gens["carrier"]).sum()

        links = _static(ds, "links", [c[len("links_"):] for c in ds.data_vars
                                      if c.startswith("links_") and not c.startswith("links_t_")
                                      and ds[c].dims == ("links_i",)])
        p0 = _series(ds, "links", "p0")
        su = _static(ds, "storage_units", ["carrier", "p_nom", "max_hours"])
        su["carrier"] = su["carrier"].astype(str)
        store = _series(ds, "storage_units", "p_store")
        dispatch = _series(ds, "storage_units", "p_dispatch")
        if store.empty:
            net = _series(ds, "storage_units", "p")
            store, dispatch = (-net).clip(lower=0.0), net.clip(lower=0.0)
        soc = _series(ds, "storage_units", "state_of_charge")
        ask = _series(ds, "links", "marginal_cost")
        stores = _static(ds, "stores", ["carrier", "e_nom"])
        e_t = _series(ds, "stores", "e")
        h2_store = [s for s in stores.index if "H2" in str(stores.at[s, "carrier"]) and s in e_t.columns]
        out = {
            "hours": float(w.sum()),
            "bus_carrier": buses.get("carrier", pd.Series(dtype=str)).astype(str),
            "scottish_bus": scotland_by_geography(buses) & buses.get("carrier", pd.Series(dtype=str)).astype(str).eq("AC"),
            "must_run_mwh": must_run,
            "storage_charge_mwh": _carrier_sum(store, su["carrier"], w),
            "storage_discharge_mwh": _carrier_sum(dispatch, su["carrier"], w),
            "storage_soc_start_mwh": (soc.iloc[0].groupby(su["carrier"].reindex(soc.columns).to_numpy()).sum()
                                      if not soc.empty else pd.Series(dtype=float)),
            "storage_soc_end_mwh": (soc.iloc[-1].groupby(su["carrier"].reindex(soc.columns).to_numpy()).sum()
                                    if not soc.empty else pd.Series(dtype=float)),
            "h2_store_start_mwh": float(e_t[h2_store].iloc[0].sum()) if h2_store else np.nan,
            "h2_store_end_mwh": float(e_t[h2_store].iloc[-1].sum()) if h2_store else np.nan,
            "h2_store_min_mwh": float(e_t[h2_store].sum(axis=1).min()) if h2_store else np.nan,
            "links": links,
            "storage_p_nom_mw": su["p_nom"].astype(float).groupby(su["carrier"]).sum(),
            "foreign_ask_hour_mwh": _foreign_ask_hours(links, ask, w),
            **{f"ic_{k}": v for k, v in interconnector_energy(p0, links, w).items()},
            **_hydrogen_links(p0, links, w),
        }
    return out


def read_stage1(wholesale_path: str | Path, links_csv: str | Path | None, links: pd.DataFrame) -> dict[str, Any]:
    """Stage 1 (day-ahead) energy of one month: generators and storage from the saved
    wholesale network, link flows from ``_wholesale_links.csv`` (the saved wholesale network
    does not carry link flows)."""
    import xarray as xr

    with xr.open_dataset(wholesale_path) as ds:
        w = _weights(ds)
        gens = _static(ds, "generators", ["carrier", "bus"])
        p = _series(ds, "generators", "p")
        cols = [c for c in p.columns if c in gens.index]
        by_gen = pd.Series(p[cols].mul(w, axis=0).sum().to_numpy(), index=cols)
        su = _static(ds, "storage_units", ["carrier"])
        net = _series(ds, "storage_units", "p")
        storage_discharge = _carrier_sum(net.clip(lower=0.0), su["carrier"].astype(str), w)
        storage_charge = _carrier_sum((-net).clip(lower=0.0), su["carrier"].astype(str), w)
    out: dict[str, Any] = {"generator_mwh": by_gen, "storage_discharge_mwh": storage_discharge,
                           "storage_charge_mwh": storage_charge, "price_hours": np.nan}
    price_csv = Path(str(wholesale_path).replace("_wholesale.nc", "_wholesale_price.csv"))
    if price_csv.exists():
        price = pd.read_csv(price_csv, index_col=0)["wholesale_price"].astype(float).to_numpy()
        out["price_hours"] = float((price * (w if len(w) == len(price) else 1.0)).sum())
    if links_csv is not None and Path(links_csv).exists():
        flows = pd.read_csv(links_csv, index_col=0)
        flows.columns = flows.columns.astype(str)
        lw = np.ones(len(flows)) if len(flows) != len(w) else w
        out.update({f"ic_{k}": v for k, v in interconnector_energy(flows, links, lw).items()})
        out.update(_hydrogen_links(flows, links, lw))
    return out


def read_month(balancing: str | Path, wholesale: str | Path | None = None, wholesale_links: str | Path | None = None,
               hydro_config: dict | None = None) -> dict[str, Any]:
    month = {"tables": dataset_energy_tables(balancing, hydro_config), "extras": read_balancing_extras(balancing)}
    month["stage1"] = (read_stage1(wholesale, wholesale_links, month["extras"]["links"])
                       if wholesale is not None else None)
    return month


# ─────────────────────────────── combining ──────────────────────────────────

def _sum_series(values) -> pd.Series:
    values = [v for v in values if v is not None and len(v)]
    return pd.concat(values).groupby(level=0).sum() if values else pd.Series(dtype=float)


def combine_months(months: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum months; return the adjusted shared tables plus the 2050 context."""
    tables = sum_energy_tables([m["tables"] for m in months])
    ex = [m["extras"] for m in months]
    bus_carrier = ex[0]["bus_carrier"]
    scottish = ex[0]["scottish_bus"]
    gens = tables["generators"].copy()
    gens["bus_carrier"] = gens["bus"].astype(str).map(bus_carrier).fillna("AC")
    non_electric = gens[gens["bus_carrier"].ne("AC")]
    tables["generators"] = gens[gens["bus_carrier"].eq("AC")].drop(columns="bus_carrier")

    ic_import = _sum_series([e["ic_import_mwh"] for e in ex])
    ic_export = _sum_series([e["ic_export_mwh"] for e in ex])
    eu_import, eu_export = tables["import_mwh"], tables["export_mwh"]
    tables["import_mwh"], tables["export_mwh"] = float(ic_import.sum()), float(ic_export.sum())

    storage = tables["storage_dispatch_mwh"].copy()
    for member, target in STORAGE_FOLD.items():
        if member in storage.index:
            storage.loc[target] = float(storage.get(target, 0.0)) + float(storage.pop(member))
    tables["storage_dispatch_mwh"] = storage

    stage1 = [m["stage1"] for m in months if m.get("stage1") is not None]
    gen_static = gens[["carrier", "bus"]]
    s1 = None
    if len(stage1) == len(months):
        s1_gen = _sum_series([s["generator_mwh"] for s in stage1])
        s1_gen = s1_gen[s1_gen.index.isin(gen_static.index[gens["bus_carrier"].eq("AC")])]
        s1 = {
            "generator_by_carrier_mwh": s1_gen.groupby(gen_static.loc[s1_gen.index, "carrier"]).sum(),
            "storage_discharge_mwh": _sum_series([s["storage_discharge_mwh"] for s in stage1]),
            "storage_charge_mwh": _sum_series([s["storage_charge_mwh"] for s in stage1]),
        }
        for key in ("ic_net_import_mwh", "ic_net_export_mwh", "electrolysis_elec_mwh", "electrolysis_h2_mwh",
                    "reformer_gas_mwh", "reformer_h2_mwh", "h2_turbine_h2_mwh", "h2_turbine_elec_mwh"):
            s1[key] = float(sum(s.get(key, np.nan) for s in stage1))
        s1["gb_price_mean"] = float(sum(s["price_hours"] for s in stage1)) / float(sum(e["hours"] for e in ex))
        s1["ic_import_mwh"] = _sum_series([s.get("ic_import_mwh") for s in stage1])
        s1["ic_export_mwh"] = _sum_series([s.get("ic_export_mwh") for s in stage1])

    # Curtailment split for resource carriers: Stage 1 spill vs BM turn-down, and Scotland's share.
    gens_ac = tables["generators"].copy()
    gens_ac["scottish"] = gens_ac["bus"].astype(str).map(scottish).fillna(False).astype(bool)
    if s1 is not None:
        s1_by_gen = _sum_series([s["generator_mwh"] for s in stage1])
        gens_ac["stage1_mwh"] = s1_by_gen.reindex(gens_ac.index).fillna(0.0)
    else:
        gens_ac["stage1_mwh"] = np.nan
    split = {}
    for carrier in RESOURCE_SPLIT_CARRIERS:
        g = gens_ac[gens_ac["carrier"].eq(carrier)]
        if g.empty:
            continue
        scot = g[g["scottish"]]
        split[carrier] = {
            "available_mwh": float(g["available_mwh"].sum()),
            "stage1_mwh": float(g["stage1_mwh"].sum()),
            "bm_mwh": float(g["dispatched_mwh"].sum()),
            "stage1_curtailment_mwh": float((g["available_mwh"] - g["stage1_mwh"]).sum()),
            "bm_turn_down_mwh": float((g["stage1_mwh"] - g["dispatched_mwh"]).sum()),
            "bm_turn_down_scotland_mwh": float((scot["stage1_mwh"] - scot["dispatched_mwh"]).sum()),
            "available_scotland_mwh": float(scot["available_mwh"].sum()),
            "bm_scotland_mwh": float(scot["dispatched_mwh"].sum()),
            "p_nom_mw": float(g["p_nom"].sum()),
        }
    context = {
        "months": len(months),
        "hours": float(sum(e["hours"] for e in ex)),
        "eu_market_import_mwh": eu_import, "eu_market_export_mwh": eu_export,
        "ic_import_by_cable_mwh": ic_import, "ic_export_by_cable_mwh": ic_export,
        "ic_net_import_mwh": float(sum(e["ic_net_import_mwh"] for e in ex)),
        "ic_net_export_mwh": float(sum(e["ic_net_export_mwh"] for e in ex)),
        "non_electric_generators_mwh": non_electric.groupby("carrier")["dispatched_mwh"].sum(),
        "must_run_mwh": _sum_series([e["must_run_mwh"] for e in ex]),
        "storage_p_nom_mw": ex[0]["storage_p_nom_mw"],
        # Electrical: a hydrogen turbine's p_nom is on its hydrogen (bus0) side.
        "h2_turbine_p_nom_mw": float((lambda lk: (pd.to_numeric(lk["p_nom"], errors="coerce")
                                                  * pd.to_numeric(lk["efficiency"], errors="coerce")).sum())(
            ex[0]["links"][ex[0]["links"]["carrier"].astype(str).eq(H2_TURBINE)])),
        "foreign_ask_mean": _sum_series([e["foreign_ask_hour_mwh"] for e in ex]) / float(sum(e["hours"] for e in ex)),
        "storage_charge_mwh": _sum_series([e["storage_charge_mwh"] for e in ex]),
        "storage_discharge_mwh": _sum_series([e["storage_discharge_mwh"] for e in ex]),
        "storage_soc_start_mwh": ex[0]["storage_soc_start_mwh"],
        "storage_soc_end_mwh": ex[-1]["storage_soc_end_mwh"],
        "h2_store_start_mwh": ex[0]["h2_store_start_mwh"], "h2_store_end_mwh": ex[-1]["h2_store_end_mwh"],
        "h2_store_min_mwh": float(np.nanmin([e["h2_store_min_mwh"] for e in ex])),
        "load_shedding_by_month_mwh": [float(m["tables"]["load_shedding_mwh"]) for m in months],
        "curtailment_split": split,
        "generators": gens_ac,
        "stage1": s1,
    }
    for key in ("electrolysis_elec_mwh", "electrolysis_h2_mwh", "reformer_gas_mwh", "reformer_h2_mwh",
                "h2_turbine_h2_mwh", "h2_turbine_elec_mwh"):
        context[key] = float(sum(e[key] for e in ex))
    return {"tables": tables, "context": context}


# ─────────────────────────────── validation ─────────────────────────────────

def _carrier_of_item(item: str) -> str:
    return str(item).split(" (")[0]


def _stage1_value(row: pd.Series, ctx: dict[str, Any]) -> float:
    s1 = ctx.get("stage1")
    if s1 is None:
        return np.nan
    carrier = _carrier_of_item(row["item"])
    members = FES2025_MEMBER_CARRIERS.get(carrier, (carrier,))
    by_carrier = s1["generator_by_carrier_mwh"]
    comp = row["component"]
    if comp == "dispatched_energy":
        if carrier == "Hydrogen":
            return s1["h2_turbine_elec_mwh"] / 1e3
        return float(by_carrier.reindex(list(members)).fillna(0.0).sum()) / 1e3
    if comp == "curtailment":
        split = ctx["curtailment_split"].get(carrier)
        return split["stage1_curtailment_mwh"] / 1e3 if split else np.nan
    if comp == "storage_discharge":
        carriers = [carrier] + [m for m, t in STORAGE_FOLD.items() if t == carrier]
        return float(s1["storage_discharge_mwh"].reindex(carriers).fillna(0.0).sum()) / 1e3
    if comp == "interconnectors" and row["item"] == "Imports":
        return float(s1["ic_import_mwh"].sum()) / 1e3
    if comp == "interconnectors" and row["item"] == "Exports":
        return float(s1["ic_export_mwh"].sum()) / 1e3
    if comp == "electrolysis":
        return s1["electrolysis_elec_mwh"] / 1e3
    return np.nan


def _extra_rows(targets: dict[str, Any], ctx: dict[str, Any], tol: dict[str, float]) -> list[dict[str, Any]]:
    """The 2050 rows the shared builder has no notion of (module note)."""
    min_abs = float(tol["min_abs_gwh"])
    rel = float(tol["dispatch_rel_tol"])
    rows: list[dict[str, Any]] = []
    s1 = ctx.get("stage1") or {}

    def add(component, item, target, model, rel_tol, note, lines="", stage1=np.nan):
        rows.append({**_row(component, item, target, model, rel_tol, False, min_abs, note, lines), "stage1_gwh": stage1})

    # Storage charging against ES1 consumption (context for the discharge rows).
    storage_t = targets.get("storage", pd.DataFrame())
    charge = ctx["storage_charge_mwh"].copy()
    s1_charge = s1.get("storage_charge_mwh", pd.Series(dtype=float)).copy()
    for member, target in STORAGE_FOLD.items():
        for series in (charge, s1_charge):
            if member in series.index:
                series.loc[target] = float(series.get(target, 0.0)) + float(series.pop(member))
    for carrier in storage_t.index:
        add("storage_charge", carrier + (" (Battery + Domestic Battery)" if carrier == "Battery" else ""),
            float(storage_t.at[carrier, "consumption_gwh"]), float(charge.get(carrier, 0.0)) / 1e3, None,
            "ES1 storage consumption; judged on the discharge row", "ES1 Storage",
            float(s1_charge.get(carrier, np.nan)) / 1e3 if len(s1_charge) else np.nan)
    for carrier in sorted(set(charge.index) - set(storage_t.index)):
        add("storage_charge", carrier, None, float(charge[carrier]) / 1e3, None,
            "model storage carrier with no ES1 line (consumer flexibility, V2G or DACCS drain)", "",
            float(s1_charge.get(carrier, np.nan)) / 1e3 if len(s1_charge) else np.nan)

    # Interconnectors: GB-hourly net, and the foreign-bus netting the shared reader uses.
    ic = targets.get("interconnectors", {})
    add("interconnectors", "Net exports (annual)", -float(ic.get("netflow_gwh", np.nan)),
        (ctx["ic_net_export_mwh"] - ctx["ic_net_import_mwh"]) / 1e3, rel,
        "ES1 export - import; model GB-side net over the year", "ES1 Interconnectors",
        (s1.get("ic_net_export_mwh", np.nan) - s1.get("ic_net_import_mwh", np.nan)) / 1e3)
    add("interconnectors", "Imports (GB net per hour)", None, ctx["ic_net_import_mwh"] / 1e3, None,
        "all cables netted each hour; a lower bound on gross trade")
    add("interconnectors", "Exports (GB net per hour)", None, ctx["ic_net_export_mwh"] / 1e3, None,
        "all cables netted each hour; a lower bound on gross trade")
    add("interconnectors", "Imports (EU_market balance generators)", None, ctx["eu_market_import_mwh"] / 1e3, None,
        "the shared CP30 reader's measure (netted per foreign bus, foreign side); not the comparison basis in 2050")
    add("interconnectors", "Exports (EU_market balance generators)", None, ctx["eu_market_export_mwh"] / 1e3, None,
        "the shared CP30 reader's measure (netted per foreign bus, foreign side); not the comparison basis in 2050")
    add("interconnectors", "Capacity (MW)", float(ic.get("capacity_mw", np.nan)), None, None,
        "ES1 interconnector capacity, MW (not GWh); see the IC1 diagnostics for the model's cables")

    # Hydrogen system against WS1 (calendar TWh H2 -> GWh).
    h2 = targets.get("hydrogen", {})
    ws1 = (h2.get("annual_twh_calendar") or h2.get("annual_twh") or {})
    supply, demand = ws1.get("supply", {}), ws1.get("demand", {})
    add("hydrogen", "Electrolysis hydrogen output (TWh H2 as GWh)",
        1e3 * sum(float(supply.get(k, 0.0)) for k in WS1_ELECTROLYSIS), ctx["electrolysis_h2_mwh"] / 1e3, rel,
        "electrolysis links: electricity in x efficiency", "WS1 Networked Electrolysis",
        s1.get("electrolysis_h2_mwh", np.nan) / 1e3)
    add("hydrogen", "Reformer (CCS) hydrogen output (TWh H2 as GWh)", 1e3 * float(supply.get(WS1_REFORMER, np.nan)),
        ctx["reformer_h2_mwh"] / 1e3, rel, "gas_reforming_ccs link: gas in x efficiency", f"WS1 {WS1_REFORMER}",
        s1.get("reformer_h2_mwh", np.nan) / 1e3)
    add("hydrogen", "Hydrogen burnt for power (TWh H2 as GWh)", 1e3 * float(demand.get(WS1_POWER, np.nan)),
        ctx["h2_turbine_h2_mwh"] / 1e3, rel, "H2_turbine links: hydrogen in", f"WS1 demand {WS1_POWER}",
        s1.get("h2_turbine_h2_mwh", np.nan) / 1e3)
    add("hydrogen", "Hydrogen store start / end of year (GWh H2)", None,
        ctx["h2_store_end_mwh"] / 1e3, None,
        f"start {ctx['h2_store_start_mwh'] / 1e6:.2f} TWh, end {ctx['h2_store_end_mwh'] / 1e6:.2f} TWh, "
        f"minimum {ctx['h2_store_min_mwh'] / 1e6:.2f} TWh (BM networks); model value is the end level")
    for carrier, value in ctx["non_electric_generators_mwh"].items():
        add("hydrogen", f"{carrier} (non-electric generator)", None, float(value) / 1e3, None,
            "natural gas into the reformer; excluded from the ES1 electricity comparison")

    # DACCS and the demand components (the loads are loss-inclusive; FES blocks are consumer basis).
    comp = targets.get("demand_components", {})
    model_lines = ctx.get("demand_components_mwh", pd.Series(dtype=float))
    add("demand_component", "DACCS electricity", float(comp.get("daccs_gwh", np.nan)),
        float(model_lines.get(DACCS_CARRIER, np.nan)) / 1e3, rel,
        "DACCS flexible load (decision 25), outside ED1 demand", "ES1 DACCS")
    for item, key, line in (("Heat pumps", "heat_pumps_gwh", "heat_pumps"), ("EVs", "evs_gwh", "evs"),
                            ("District heat (residential)", "district_heat_residential_gwh", "district_heat"),
                            ("Data centres", "data_centres_gwh", "data_centres"),
                            ("Residual", None, "residual")):
        # FES's residual (ED1 consumer total less heat pumps and EVs) still holds data centres and
        # district heat, which the model carries as their own lines: no like-for-like target.
        add("demand_component", item, float(comp.get(key, np.nan)) if key in comp else None,
            float(model_lines.get(line, np.nan)) / 1e3, None,
            "INFO: FES blocks are consumer basis; the model's loads are system basis (losses in), so the "
            "total, not each line, is judged (the demand row)", "FES Dem_BB / ED1")

    # Curtailment split for every resource carrier.
    for carrier, s in ctx["curtailment_split"].items():
        add("curtailment_split", f"{carrier}: Stage 1 spill", None, s["stage1_curtailment_mwh"] / 1e3, None,
            "available - Stage 1 dispatch (copperplate surplus)")
        add("curtailment_split", f"{carrier}: BM turn-down", None, s["bm_turn_down_mwh"] / 1e3, None,
            "Stage 1 - BM dispatch (network constraints)")
        add("curtailment_split", f"{carrier}: BM turn-down in Scotland", None,
            s["bm_turn_down_scotland_mwh"] / 1e3, None, "buses north of the border (scotland_by_geography)")

    # Load shedding by month.
    for month, value in zip(MONTHS, ctx["load_shedding_by_month_mwh"]):
        add("load_shedding_by_month", month, None, value / 1e3, None, "BM unserved energy")
    return rows


def validate(targets: dict[str, Any], combined: dict[str, Any],
             tolerances: dict[str, float] | None = None) -> pd.DataFrame:
    """The ES1 reconciliation table, with Stage 1 values, the 2050 rows and explanations."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    tables, ctx = combined["tables"], combined["context"]
    ctx["demand_components_mwh"] = tables.get("demand_components_mwh", pd.Series(dtype=float))
    table = build_energy_validation(targets, tables, stage="dispatch", tolerances=tol)
    target = target_column(table)
    table["stage1_gwh"] = [_stage1_value(row, ctx) for _, row in table.iterrows()]
    table.loc[(table["component"] == "storage_discharge") & table["item"].eq("Battery"), "item"] = \
        "Battery (Battery + Domestic Battery)"
    ic = table["component"].eq("interconnectors")
    table.loc[ic, "note"] = "per cable, GB side: imports delivered, exports given up by GB (B9); summed over cables"
    extra = pd.DataFrame(_extra_rows(targets, ctx, tol)).rename(columns={"cp30_gwh": target})
    extra.insert(0, "stage", "dispatch")
    table = pd.concat([table, extra], ignore_index=True)
    table["explanation"] = [explain(row, ctx, targets) if row["status"] == "WARN" else ""
                            for _, row in table.iterrows()]
    return table


# ─────────────────────────────── explanations ───────────────────────────────
#
# Each WARN gets a sentence built from the run's own numbers. The causes are those found on
# the decision-40 year (2 October 2026); after a re-solve every explanation must be re-read,
# and a WARN with no explainer fails the run.

def _twh(mwh: float) -> str:
    return f"{mwh / 1e6:,.1f}"


def _target_twh(row) -> str:
    target = row.get("target_gwh", row.get("cp30_gwh"))
    return f"{float(target) / 1e3:,.1f}"


def _gen_ctx(ctx, carriers) -> dict[str, float]:
    gens = ctx["generators"]
    g = gens[gens["carrier"].isin(carriers)]
    hours = ctx["hours"]
    return {"p_nom": float(g["p_nom"].sum()), "available": float(g["available_mwh"].sum()),
            "bm": float(g["dispatched_mwh"].sum()), "s1": float(g["stage1_mwh"].sum()),
            "lf": float(g["dispatched_mwh"].sum()) / (float(g["p_nom"].sum()) * hours) if len(g) else np.nan,
            "must_run": float(ctx["must_run_mwh"].reindex(list(carriers)).fillna(0.0).sum())}


def _explain_curtailment(row, ctx, targets) -> str:
    carrier = _carrier_of_item(row["item"])
    s = ctx["curtailment_split"][carrier]
    model_total = sum(v["available_mwh"] - v["bm_mwh"] for k, v in ctx["curtailment_split"].items()
                      if k in CURTAILMENT_CARRIERS)
    gen = targets["generation"]
    fes_total = float(gen.reindex(list(CURTAILMENT_CARRIERS))["curtailment_gwh"].fillna(0.0).sum()) * 1e3
    text = (f"Available {_twh(s['available_mwh'])} TWh against ES1 gross {float(gen.at[carrier, 'gross_gwh']) / 1e3:,.1f} "
            f"(calibrated, so the gap is curtailment, a model result). Stage 1 spills "
            f"{_twh(s['stage1_curtailment_mwh'])} TWh (copperplate surplus); the BM turns down "
            f"{_twh(s['bm_turn_down_mwh'])} TWh more, {_twh(s['bm_turn_down_scotland_mwh'])} of it in Scotland. ")
    if row["difference_gwh"] > 0:
        text += ("The BM turn-down is the network: boundaries held at their 2045 capability (decisions 10 and 36) "
                 "against 2050 demand, so northern wind is replaced south of the constraints. ")
    else:
        text += ("The model spills less of this carrier than FES and more wind; ")
    text += (f"Wind and solar curtailment together: model {_twh(model_total)} TWh against FES {_twh(fes_total)} TWh. "
             "FES's curtailment comes from its own network model (decision 45 for onshore).")
    return text


def _explain_dispatch(row, ctx, targets) -> str:
    carrier = _carrier_of_item(row["item"])
    s1 = ctx.get("stage1") or {}
    members = FES2025_MEMBER_CARRIERS.get(carrier, (carrier,))
    g = _gen_ctx(ctx, members)
    target_gwh = float(row.get("target_gwh", row.get("cp30_gwh")))
    cap_t = float(targets["generation"].at[carrier, "capacity_mw"]) if carrier in targets["generation"].index else np.nan
    base = (f"Model {g['bm'] / 1e6:,.1f} TWh (Stage 1 {g['s1'] / 1e6:,.1f}) on {g['p_nom'] / 1e3:,.2f} GW, load factor "
            f"{g['lf']:.2f}; ES1 {target_gwh / 1e3:,.1f} TWh on {cap_t / 1e3:,.2f} GW "
            f"(load factor {target_gwh * 1e3 / (cap_t * ctx['hours']) if cap_t else np.nan:.2f}). ")
    if carrier in ("wind_onshore", "wind_offshore", "solar_pv"):
        s = ctx["curtailment_split"][carrier]
        return base + (f"Availability matches ES1 gross; the shortfall is curtailment: "
                       f"{_twh(s['stage1_curtailment_mwh'])} TWh in Stage 1 and {_twh(s['bm_turn_down_mwh'])} in the BM "
                       f"({_twh(s['bm_turn_down_scotland_mwh'])} in Scotland, behind boundaries held at 2045: decisions 36 "
                       "and 45).")
    if carrier == "CCS Gas":
        split = ctx["curtailment_split"]
        renewables_down = sum(v["bm_turn_down_mwh"] for v in split.values())
        exports_down = (float(s1.get("ic_export_mwh", pd.Series(dtype=float)).sum())
                        - float(ctx["ic_export_by_cable_mwh"].sum()))
        h2_up = ctx["h2_turbine_elec_mwh"] - s1.get("h2_turbine_elec_mwh", np.nan)
        return base + (f"In Stage 1 (copperplate) gas CCS runs {g['s1'] / 1e6:,.1f} TWh, below ES1. The BM adds "
                       f"{(g['bm'] - g['s1']) / 1e6:,.1f} TWh: it turns down {_twh(renewables_down)} TWh of renewables "
                       f"behind boundaries held at their 2045 capability (decision 36) and replaces them south of the "
                       f"boundaries with gas CCS (66 GBP/MWh), {_twh(h2_up)} TWh more hydrogen-turbine output, and "
                       f"{_twh(exports_down)} TWh less export. So the gap is the network, not the fleet or the merit "
                       "order: FES's own network carries more of the northern wind south.")
    if carrier == "Hydrogen":
        h2_gw = float(ctx["h2_turbine_p_nom_mw"]) / 1e3
        base = (f"Hydrogen turbines (links) give {_twh(ctx['h2_turbine_elec_mwh'])} TWh (Stage 1 "
                f"{_twh(s1.get('h2_turbine_elec_mwh', np.nan))}) on {h2_gw:,.1f} GW, load factor "
                f"{ctx['h2_turbine_elec_mwh'] / (h2_gw * 1e3 * ctx['hours']):.3f} electrical; ES1 "
                f"{float(targets['generation'].at['Hydrogen', 'generation_gwh']) / 1e3:,.1f} TWh on {cap_t / 1e3:,.1f} GW. ")
        return base + (f"Hydrogen burnt {_twh(ctx['h2_turbine_h2_mwh'])} TWh_H2 (Stage 1 "
                       f"{_twh(s1.get('h2_turbine_h2_mwh', np.nan))}) against WS1's power-generation "
                       f"demand. Gas CCS (66 GBP/MWh SRMC) dispatches ahead of the turbines' DPA price "
                       f"(reference CCGT - 1, decision 38), and the BM keeps the Stage 1 store band (decision 40) and "
                       f"the annual pass a 4 TWh reserve, so hydrogen is held for winter: the store runs "
                       f"{_twh(ctx['h2_store_start_mwh'])} -> {_twh(ctx['h2_store_end_mwh'])} TWh, minimum "
                       f"{_twh(ctx['h2_store_min_mwh'])}.")
    if carrier == "biomass":
        return base + (f"Unabated biomass has a 0.4 minimum stable output in the model (must-run floor "
                       f"{_twh(g['must_run'])} TWh), which ES1's 0.06 load factor does not have. Small in absolute "
                       "terms; a fleet-data item, not a dispatch result.")
    if carrier == "large_hydro":
        s = ctx["curtailment_split"]["large_hydro"]
        return base + (f"Inflow (available) {_twh(g['available'])} TWh matches ES1, and Stage 1 turbines all of it "
                       f"({_twh(s['stage1_mwh'])}). The BM turns down {_twh(s['bm_turn_down_mwh'])} TWh, "
                       f"{_twh(s['bm_turn_down_scotland_mwh'])} of it in Scotland: the Highland reservoirs sit behind "
                       "the same boundaries as the northern wind (decision 36).")
    if carrier == "marine":
        return base + (f"Marine availability {_twh(g['available'])} TWh is above ES1's {target_gwh / 1e3:,.1f} "
                       "(no ES1 curtailment line), so the gap is the tidal-stream profile's capacity factor, a "
                       "build-time calibration item; dispatch itself takes nearly all of it.")
    if carrier == "gas":
        # 2040 (23 Nov 2026): unabated gas, absent from 2050.
        moved = (g["bm"] - g["s1"]) / 1e6
        if g["s1"] > target_gwh * 1e3:
            return base + (f"Stage 1 (copperplate, before any network constraint) already runs {g['s1'] / 1e6:,.1f} TWh "
                           f"against ES1's {target_gwh / 1e3:,.1f}, so the gap is in the merit order, not balancing (the "
                           f"BM moves it by {moved:+,.1f} TWh). Unabated gas is the last fossil plant in the stack and runs "
                           "in the hours nothing cheaper is left; ES1 has far more storage discharge and hydrogen-turbine "
                           "output than the model in the same year (their lines), which is where FES serves those hours.")
        return base + (f"Stage 1 runs {g['s1'] / 1e6:,.1f} TWh and the BM moves it by {moved:+,.1f} TWh, replacing "
                       "plant turned down behind the boundaries.")
    if carrier == "CHP":
        # 2040 (23 Nov 2026): gas CHP, absent from 2050.
        floor = (f"no heat-led must-run (floor {_twh(g['must_run'])} TWh)" if g["must_run"] < 0.05 * target_gwh * 1e3
                 else f"a must-run floor of {_twh(g['must_run'])} TWh")
        return base + (f"The model runs gas CHP on its electricity merit order with {floor}: available "
                       f"{_twh(g['available'])} TWh, Stage 1 {_twh(g['s1'])}, BM {_twh(g['bm'])}. ES1 runs CHP for its "
                       "heat as well, which the model does not represent. Small in absolute terms.")
    if carrier == "waste_to_energy":
        return base + (f"Waste and biogas are priced (waste 140, biogas 81 GBP/MWh) and dispatch on merit: "
                       f"available {_twh(g['available'])} TWh, run {_twh(g['bm'])}. FES runs them near baseload. "
                       "With a year-mean Stage 1 price near 30 GBP/MWh, merit order cuts their output.")
    return ""


def _explain_storage(row, ctx, targets) -> str:
    carrier = _carrier_of_item(row["item"])
    storage_t = targets["storage"]
    cap = float(storage_t.at[carrier, "capacity_mw"]) if carrier in storage_t.index else np.nan
    carriers = [carrier] + [m for m, t in STORAGE_FOLD.items() if t == carrier]
    s1 = ctx.get("stage1") or {}
    s1_dis = float(s1.get("storage_discharge_mwh", pd.Series(dtype=float)).reindex(carriers).fillna(0.0).sum())
    bm = float(row["model_gwh"]) * 1e3
    target = float(row.get("target_gwh", row.get("cp30_gwh"))) * 1e3
    v2g = float(ctx["storage_discharge_mwh"].get("V2G_fleet", 0.0))
    v2g_gw = float(ctx["storage_p_nom_mw"].get("V2G_fleet", 0.0)) / 1e3
    return (f"ES1 cycles this storage {target / cap:,.0f} full-power hours a year; the model {bm / cap:,.0f} "
            f"(Stage 1 {s1_dis / 1e6:,.1f} TWh, BM {bm / 1e6:,.1f} TWh). The day-ahead windows see a flat, low price "
            "much of the year (curtailment hours near zero, hydrogen-set hours near the DPA price), so daily "
            f"arbitrage spreads are thin; V2G ({v2g_gw:,.0f} GW, {_twh(v2g)} TWh discharged) and consumer flexibility take the "
            "same spreads and ES1's storage lines exclude them. The BM anchors storage to Stage 1 (gross-leg anchors). "
            "A model result: FES's BID3 cycles storage against its own price shape.")


def _explain_interconnectors(row, ctx, targets) -> str:
    s1 = ctx.get("stage1") or {}
    item = row["item"]
    imp, exp = float(ctx["ic_import_by_cable_mwh"].sum()), float(ctx["ic_export_by_cable_mwh"].sum())
    asks = ", ".join(f"{k} {v:,.1f}" for k, v in ctx["foreign_ask_mean"].items())
    return (f"Model imports {_twh(imp)} and exports {_twh(exp)} TWh (Stage 1 "
            f"{_twh(float(s1.get('ic_import_mwh', pd.Series(dtype=float)).sum()))} / "
            f"{_twh(float(s1.get('ic_export_mwh', pd.Series(dtype=float)).sum()))}); ES1 "
            f"{targets['import_gwh'] / 1e3:,.1f} / {targets['export_gwh'] / 1e3:,.1f}. Foreign prices are TYNDP 2024 "
            "Distributed Energy 2050 with IC1's tranches (decision 27), not FES's own European dispatch. Mean "
            f"import asks {asks} GBP/MWh against a GB Stage 1 mean of {s1.get('gb_price_mean', np.nan):,.1f}, so GB "
            "exports in its surplus hours and imports in the scarce ones. The BM cuts exports by "
            f"{_twh(float(s1.get('ic_export_mwh', pd.Series(dtype=float)).sum()) - exp)} TWh (counter-trading what "
            "the network cannot carry to the southern cables). See the IC1 diagnostics for the cable view. "
            + ("The net position is the same sign as ES1 (GB a net exporter)." if "Net" in item else ""))


def _explain_electrolysis(row, ctx, targets) -> str:
    return (f"Electrolysis takes {_twh(ctx['electrolysis_elec_mwh'])} TWh against ES1's "
            f"{targets['electrolysis_gwh'] / 1e3:,.1f}: the annual hydrogen balance is met with more reforming "
            f"({_twh(ctx['reformer_h2_mwh'])} TWh_H2) when electricity is not surplus.")


def _explain_hydrogen(row, ctx, targets) -> str:
    item = row["item"]
    if item.startswith("Hydrogen burnt"):
        return _explain_dispatch(pd.Series({**row.to_dict(), "item": "Hydrogen",
                                            "target_gwh": float(targets["generation"].at["Hydrogen", "generation_gwh"])}),
                                 ctx, targets)
    if item.startswith("Reformer"):
        return (f"The reformer (5.38 GW_H2, gas plus CCS cost) makes {_twh(ctx['reformer_h2_mwh'])} TWh_H2: it is "
                "the marginal hydrogen all year in the annual pass (hydrogen value 36.75-40.5 GBP/MWh), so it runs "
                "whenever electrolysis is not cheaper, and the hydrogen balance (decision 15) fixes the sum.")
    if item.startswith("Electrolysis"):
        return _explain_electrolysis(row, ctx, targets)
    return ""


def _explain_demand_component(row, ctx, targets) -> str:
    return ""


EXPLAINERS: dict[str, Callable[[pd.Series, dict, dict], str]] = {
    "dispatched_energy": _explain_dispatch,
    "curtailment": _explain_curtailment,
    "storage_discharge": _explain_storage,
    "interconnectors": _explain_interconnectors,
    "electrolysis": _explain_electrolysis,
    "hydrogen": _explain_hydrogen,
    "demand_component": _explain_demand_component,
}


def explain(row: pd.Series, ctx: dict[str, Any], targets: dict[str, Any]) -> str:
    explainer = EXPLAINERS.get(row["component"])
    text = explainer(row, ctx, targets) if explainer else ""
    return text or "UNEXPLAINED: add an explanation in scripts/FES/fes_dispatch_validation.py"


def unexplained(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["status"].eq("WARN") & table["explanation"].str.startswith("UNEXPLAINED")]


# ─────────────────────────────── entry points ───────────────────────────────

def summary_json(table: pd.DataFrame, combined: dict[str, Any], study: str, pathway: str, year: int) -> dict:
    ctx = combined["context"]
    return {
        "study": study, "pathway": pathway, "modelled_year": year, "target_source": "FES2025",
        "months": ctx["months"], "hours": ctx["hours"],
        "status_counts": table["status"].value_counts().to_dict(),
        "unexplained_warns": int(len(unexplained(table))),
        "curtailment_split_gwh": {c: {k: v / 1e3 for k, v in s.items() if k.endswith("_mwh")}
                                  for c, s in ctx["curtailment_split"].items()},
        "interconnectors_by_cable_gwh": {
            "import": (ctx["ic_import_by_cable_mwh"] / 1e3).round(3).to_dict(),
            "export": (ctx["ic_export_by_cable_mwh"] / 1e3).round(3).to_dict()},
        "rows": json.loads(table.to_json(orient="records")),
    }


def run(study: str, balancing: list[str], wholesale: list[str] | None, wholesale_links: list[str] | None,
        workbook: str, pathway: str, modelled_year: int, output_csv: str, output_json: str,
        hydro_config: dict | None = None, tolerances: dict | None = None, logger=None) -> pd.DataFrame:
    import logging

    logger = logger or logging.getLogger("fes_dispatch_validation")
    wholesale = wholesale or [None] * len(balancing)
    wholesale_links = wholesale_links or [None] * len(balancing)
    months = []
    for bal, ws, wl in zip(balancing, wholesale, wholesale_links):
        months.append(read_month(bal, ws, wl, hydro_config))
        logger.info("Read %s (%.0f h)", Path(bal).name, months[-1]["extras"]["hours"])
    combined = combine_months(months)
    if abs(combined["context"]["hours"] - hours_in_year(modelled_year)) > 0.5:
        logger.warning("Solved months cover %.0f h, not calendar %s (%d h)", combined["context"]["hours"],
                       modelled_year, hours_in_year(modelled_year))
    targets = energy_targets("FES2025", workbook, pathway, modelled_year, year_basis="calendar",
                             demand_basis="system", include_building_blocks=True)
    table = validate(targets, combined, tolerances)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    Path(output_json).write_text(json.dumps(summary_json(table, combined, study, pathway, modelled_year), indent=2,
                                            default=float), encoding="utf-8")
    logger.info("%s final-dispatch validation against FES 2025 ES1 (%s %s):\n%s", study, pathway, modelled_year,
                format_table(table))
    failures = energy_validation_failures(table)
    missing = unexplained(table)
    if not failures.empty or not missing.empty:
        raise ValueError(f"{study} dispatch energy validation failed:\n"
                         + pd.concat([failures, missing])[["component", "item", "status", "note"]].to_string(index=False))
    return table


def format_table(table: pd.DataFrame) -> str:
    target = target_column(table)
    view = table[["component", "item", target, "stage1_gwh", "model_gwh", "relative_difference", "status"]].copy()
    for col in (target, "stage1_gwh", "model_gwh"):
        view[col] = view[col] / 1e3
    view = view.rename(columns={target: "target_TWh", "stage1_gwh": "stage1_TWh", "model_gwh": "model_TWh",
                                "relative_difference": "rel"})
    return view.to_string(index=False, float_format=lambda v: f"{v:,.3f}")


def _paths(study: str, months=MONTHS, family: str = "market") -> tuple[list[str], list[str], list[str]]:
    base = Path("resources/market")
    names = [f"{study}_{m}_{family}" for m in months]
    return ([str(base / f"{n}_balancing.nc") for n in names], [str(base / f"{n}_wholesale.nc") for n in names],
            [str(base / f"{n}_wholesale_links.csv") for n in names])


if "snakemake" in globals():
    from scripts.utilities.logging_config import setup_logging

    _cfg = dict(snakemake.params.scenario_config)
    run(study=snakemake.wildcards.study, balancing=list(snakemake.input.balancing),
        wholesale=list(snakemake.input.wholesale), wholesale_links=list(snakemake.input.wholesale_links),
        workbook=snakemake.input.workbook, pathway=_cfg["FES_scenario"],
        modelled_year=int(_cfg["modelled_year"]), output_csv=snakemake.output.csv,
        output_json=snakemake.output.json, hydro_config=_cfg.get("renewables", {}).get("hydro", {}),
        tolerances=(_cfg.get("cp30_validation", {}) or {}).get("energy"),
        logger=setup_logging(snakemake.log[0] if snakemake.log else "fes_dispatch_validation"))
elif __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, help="the market year's study, e.g. FES25_HT_2050")
    ap.add_argument("--family", default="market", help="the month family: months are <study>_<Mon>_<family>")
    ap.add_argument("--workbook", default=FES2025_WORKBOOK)
    ap.add_argument("--pathway", default="Holistic Transition")
    ap.add_argument("--modelled-year", type=int, required=True)
    ap.add_argument("--inflow-profile-factor", type=float, default=0.55)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", default=None)
    a = ap.parse_args()
    bal, ws, wl = _paths(a.study, family=a.family)
    hydro = {"enabled": True, "large_hydro": {"reservoir_model": True, "inflow_profile_factor": a.inflow_profile_factor}}
    run(a.study, bal, ws, wl, a.workbook, a.pathway, a.modelled_year, a.output_csv,
        a.output_json or str(Path(a.output_csv).with_suffix(".json")), hydro)
