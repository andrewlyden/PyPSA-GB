"""Scarcity pricing in the Stage 1 wholesale solve (item MR3(b)).

The copperplate wholesale LP clears at the marginal unit's short-run cost, so
with ample capacity the price can never rise above one plant's cost, however
tight the system is. Real GB prices rise as the de-rated margin falls. DESNZ's
Dynamic Dispatch Model prices this as a mark-up over short-run cost "derived
from historic data" and reflecting "reduced capacity margins"; its peer review
asks that tightness be measured against capacity actually available.

This module does that inside the LP, as a stepped reserve shortage curve:

* **Margin.** De-rated capacity less the demand it serves, as NESO computes
  the de-rated margin for the BSC loss of load probability: conventional
  plant at its available capacity times a fuel-type de-rating factor (the
  Capacity Market factors NESO publishes), wind and other variable renewables
  at their available output (the BSC uses the wind forecast, not a de-rated
  capacity), load shedding and foreign plant excluded. In the LP this is
  ``sum_g (d_g x available_g - p_g)`` over GB generators: dispatch uses up
  margin one for one. Storage and interconnectors enter in one of two modes
  (``margin_definition`` in the curve file):

  - ``flow``: by what they actually do. Discharging and importing displace GB
    generation and so relieve the margin; charging and exporting add to the
    demand GB plant must serve and so use it up. The BSC counts
    interconnector exports as demand and imports at their notified flow.
  - ``derated_capacity``: at a Capacity Market de-rating factor on their
    capacity (storage by duration, each interconnector its own), less their
    delivery (discharge, gross import), so charging and exporting still count
    as demand.
* **Shortage curve.** Below a set of thresholds the LP buys shortfall at a
  rising price per MWh. Because every MW of demand uses up a MW of margin,
  the price of the deepest step in use becomes part of the energy price,
  which is the scarcity mark-up.
* **Calibration.** Thresholds and prices are declared in NESO de-rated
  margin terms (``data/market/scarcity/scarcity_curve_2024.json``, written by
  ``scripts/market/calibrate_scarcity_curve.py``) from the 2024 gap between
  actual N2EX prices and the model, binned on NESO's published de-rated
  margin. A linear map fitted on 2024 converts them to the model's margin.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

GB_COUNTRY_CODES = {"", "gb", "uk", "united kingdom", "great britain", "nan", "none"}
DEFAULT_EXCLUDED_CARRIERS = ("load_shedding", "EU_market", "EU_import")


WINTER_MONTHS = (10, 11, 12, 1, 2, 3)


def season_of(timestamp) -> str:
    """NESO's reporting seasons: winter October to March, summer April to September."""
    return "winter" if pd.Timestamp(timestamp).month in WINTER_MONTHS else "summer"


SEASONS = ("winter", "summer")


def load_scarcity_curve(path, max_drm_threshold_mw: float | None = None) -> dict:
    """Read the declared curve and return it in model-margin units.

    Returns ``{"requirement_mw", "steps": [(width_mw or None, price)], ...}``.
    Steps run from the top threshold downwards; the last step is unbounded.
    Thresholds are declared in NESO de-rated margin terms and converted to the
    model's margin by ``drm_from_model_margin``, in one of three forms:

    * ``model_thresholds_mw_by_season``: the model-margin value of every
      threshold in each season, set by matching 2024 hours (the model has as
      many hours below each threshold as NESO had), which a straight line
      cannot do in both the bulk and the tail;
    * ``slope`` with ``intercept_mw_by_season``: a line per season with a
      common slope (NESO's summer margin reflects planned maintenance the
      model's availability does not);
    * ``slope`` with ``intercept_mw``: one line.

    With seasonal forms ``requirement_mw`` and step widths are ``{season: MW}``.

    ``max_drm_threshold_mw`` drops every step whose threshold is above it (the
    sensitivity without the shallow bands); the remaining steps keep their
    thresholds and prices.
    """
    curve = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = dict(curve["drm_from_model_margin"])
    keep = [k for k, s in enumerate(curve["steps"])
            if max_drm_threshold_mw is None or float(s["drm_below_mw"]) <= float(max_drm_threshold_mw)]
    if not keep:
        raise ValueError(f"max_drm_threshold_mw={max_drm_threshold_mw} drops every scarcity step")
    if "model_thresholds_mw_by_season" in mapping:
        mapping["model_thresholds_mw_by_season"] = {
            season: [values[k] for k in keep] for season, values in mapping["model_thresholds_mw_by_season"].items()}
    steps_kept = [curve["steps"][k] for k in keep]
    thresholds = [float(s["drm_below_mw"]) for s in steps_kept]
    prices = [float(s["price_gbp_per_mwh"]) for s in steps_kept]
    if thresholds != sorted(thresholds, reverse=True):
        raise ValueError("scarcity steps must be ordered from the highest de-rated margin down")
    if prices != sorted(prices):
        raise ValueError("scarcity step prices must rise as the margin falls")
    if "model_thresholds_mw_by_season" in mapping:
        model = {season: [float(v) for v in values]
                 for season, values in mapping["model_thresholds_mw_by_season"].items()}
        if any(len(values) != len(thresholds) for values in model.values()):
            raise ValueError("model_thresholds_mw_by_season needs one value per step")
    else:
        slope = float(mapping["slope"])
        if slope <= 0:
            raise ValueError("drm_from_model_margin slope must be positive")
        by_season = mapping.get("intercept_mw_by_season")
        intercepts = ({season: float(v) for season, v in by_season.items()} if by_season
                      else {None: float(mapping.get("intercept_mw", 0.0))})
        # A de-rated margin threshold D corresponds to model margin (D - b) / a.
        model = {season: [(d - b) / slope for d in thresholds] for season, b in intercepts.items()}
    if None not in model and set(model) != set(SEASONS):
        raise ValueError("seasonal maps must give winter and summer")
    widths = {}
    for season, values in model.items():
        widths[season] = [values[k] - values[k + 1] for k in range(len(values) - 1)]
        if any(w <= 0 for w in widths[season]):
            raise ValueError("scarcity step widths must be positive in model units")
    seasonal = None not in model
    steps = []
    for k, price in enumerate(prices):
        if k + 1 < len(thresholds):
            width = {season: widths[season][k] for season in model} if seasonal else widths[None][k]
            steps.append((width, price))
        else:
            steps.append((None, price))
    requirement = {season: values[0] for season, values in model.items()} if seasonal else model[None][0]
    return {"requirement_mw": requirement, "steps": steps,
            "drm_thresholds_mw": thresholds, "prices": prices, "source": str(path),
            "max_drm_threshold_mw": max_drm_threshold_mw,
            "margin_definition": parse_margin_definition(curve.get("margin_definition"))}


def requirement_series(curve: dict, snapshots) -> np.ndarray:
    """The model-margin requirement in each snapshot (seasonal when declared)."""
    return seasonal_series(curve["requirement_mw"], snapshots)


def seasonal_series(value, snapshots) -> np.ndarray:
    """A scalar, or a ``{season: value}`` dict, as one value per snapshot."""
    if isinstance(value, dict):
        return np.array([value[season_of(t)] for t in snapshots], dtype=float)
    return np.full(len(snapshots), float(value))


MARGIN_MODES = ("flow", "derated_capacity")


def parse_margin_definition(definition: dict | None) -> dict:
    """De-rating factors and storage/interconnector modes; no definition = all at 1, by flow."""
    definition = definition or {}
    storage = definition.get("storage") or {}
    interconnectors = definition.get("interconnectors") or {}
    out = {
        "generator_factors": {str(k): float(v) for k, v in (definition.get("generator_factors") or {}).items()},
        "default_generator_factor": float(definition.get("default_generator_factor", 1.0)),
        "storage_mode": storage.get("mode", "flow"),
        "storage_factors_by_duration_h": sorted((float(h), float(f)) for h, f in
                                                (storage.get("factors_by_duration_h") or {}).items()),
        "interconnector_mode": interconnectors.get("mode", "flow"),
        "interconnector_factors": {str(k): float(v) for k, v in (interconnectors.get("factors") or {}).items()},
        "default_interconnector_factor": float(interconnectors.get("default_factor", 1.0)),
    }
    for key in ("storage_mode", "interconnector_mode"):
        if out[key] not in MARGIN_MODES:
            raise ValueError(f"margin_definition {key} must be one of {MARGIN_MODES}, not {out[key]!r}")
    if out["storage_mode"] == "derated_capacity" and not out["storage_factors_by_duration_h"]:
        raise ValueError("storage mode derated_capacity needs factors_by_duration_h")
    factors = list(out["generator_factors"].values()) + [f for _, f in out["storage_factors_by_duration_h"]] \
        + list(out["interconnector_factors"].values())
    if any(not 0.0 <= f <= 1.0 for f in factors):
        raise ValueError("de-rating factors must lie in [0, 1]")
    return out


def storage_derating_factor(max_hours: float, table) -> float:
    """The factor of the longest listed duration not exceeding ``max_hours`` (CM practice)."""
    if not table:
        return 1.0
    eligible = [f for h, f in table if h <= float(max_hours) + 1e-9]
    return eligible[-1] if eligible else 0.0


def interconnector_derating_factor(name: str, definition: dict) -> float:
    """Each interconnector's factor, matched on a name fragment (longest match wins)."""
    matches = [(len(k), f) for k, f in definition["interconnector_factors"].items() if k.lower() in str(name).lower()]
    return max(matches)[1] if matches else definition["default_interconnector_factor"]


def margin_factors(network, members: dict, definition: dict) -> dict:
    """Per-member de-rating factors as Series (generators, storage, import links)."""
    gens = network.generators.loc[members["generators"], "carrier"]
    gen_factor = gens.map(definition["generator_factors"]).fillna(definition["default_generator_factor"])
    su = network.storage_units.loc[members["storage"]]
    hours = su["max_hours"] if "max_hours" in su.columns else pd.Series(1.0, index=su.index)
    storage_factor = hours.map(lambda h: storage_derating_factor(h, definition["storage_factors_by_duration_h"]))
    ic_factor = pd.Series({name: interconnector_derating_factor(name, definition)
                           for name in members["import_links"]}, dtype=float)
    return {"generators": gen_factor.astype(float), "storage": storage_factor.astype(float),
            "import_links": ic_factor}


def _import_capacity(network, members: dict, snapshots) -> pd.DataFrame:
    """Import capacity into GB per link and snapshot (MW)."""
    names = list(members["import_links"])
    if not names:
        return pd.DataFrame(index=snapshots)
    p_max = network.get_switchable_as_dense("Link", "p_max_pu", snapshots)[names]
    p_min = network.get_switchable_as_dense("Link", "p_min_pu", snapshots)[names]
    cap_pu = pd.DataFrame({name: (p_max[name] if members["import_links"][name] > 0 else -p_min[name])
                           for name in names})
    return cap_pu.mul(network.links.loc[names, "p_nom"], axis=1).clip(lower=0.0)


def derated_capacity(network, members: dict, definition: dict, snapshots) -> pd.Series:
    """The constant part of the margin: de-rated available capacity per snapshot (MW)."""
    factors = margin_factors(network, members, definition)
    total = pd.Series(0.0, index=snapshots)
    gens = members["generators"]
    if gens:
        avail = network.get_switchable_as_dense("Generator", "p_max_pu", snapshots)[gens] \
            .mul(network.generators.loc[gens, "p_nom"], axis=1)
        total += avail.mul(factors["generators"], axis=1).sum(axis=1)
    if definition["storage_mode"] == "derated_capacity" and members["storage"]:
        units = members["storage"]
        p_max = network.get_switchable_as_dense("StorageUnit", "p_max_pu", snapshots)[units] \
            .mul(network.storage_units.loc[units, "p_nom"], axis=1)
        total += p_max.mul(factors["storage"], axis=1).sum(axis=1)
    if definition["interconnector_mode"] == "derated_capacity" and members["import_links"]:
        total += _import_capacity(network, members, snapshots).mul(factors["import_links"], axis=1).sum(axis=1)
    return total


def _foreign_buses(network) -> set:
    buses = network.buses
    foreign = set()
    country = buses["country"] if "country" in buses.columns else pd.Series("", index=buses.index)
    for name, value in country.items():
        text = str(value).strip().lower()
        if text not in GB_COUNTRY_CODES or str(name).lower().startswith(("external_", "hvdc_external")):
            foreign.add(name)
    return foreign


def margin_members(network, excluded_carriers=DEFAULT_EXCLUDED_CARRIERS) -> dict:
    """GB generators, storage units and import links that make up the margin."""
    foreign = _foreign_buses(network)
    gens = network.generators
    generators = gens.index[~gens["carrier"].isin(excluded_carriers) & ~gens["bus"].isin(foreign)]
    # Storage abroad (the coupled European zones, item F13) is not GB margin.
    storage = network.storage_units.index[~network.storage_units["bus"].isin(foreign)]
    links = network.links
    into_gb = {}
    for name, row in links.iterrows():
        f0, f1 = row["bus0"] in foreign, row["bus1"] in foreign
        if f0 and not f1:
            into_gb[name] = 1.0      # positive p flows into GB
        elif f1 and not f0 and float(row.get("p_min_pu", 0.0)) < 0.0:
            into_gb[name] = -1.0     # negative p flows into GB (a one-way export link cannot)
    return {"generators": list(generators), "storage": list(storage), "import_links": into_gb}


def model_margin_ex_post(network, dispatch: dict, definition: dict | None = None,
                         excluded_carriers=DEFAULT_EXCLUDED_CARRIERS) -> pd.Series:
    """The same margin from a solved dispatch, for calibration and reporting.

    ``dispatch`` holds ``generators_p`` and, when storage or interconnectors are
    counted at de-rated capacity, ``storage_dispatch`` (discharge, MW; or
    ``storage_p`` net, whose positive part is used) and ``links_p``, all indexed
    by snapshot. ``definition`` is a parsed ``margin_definition`` (None = all
    factors 1, storage and interconnectors by flow).
    """
    definition = definition if definition is not None else parse_margin_definition(None)
    members = margin_members(network, excluded_carriers)
    snapshots = dispatch["generators_p"].index
    margin = derated_capacity(network, members, definition, snapshots)
    gens = members["generators"]
    if gens:
        margin -= dispatch["generators_p"].reindex(columns=gens).fillna(0.0).sum(axis=1)
    if definition["storage_mode"] == "derated_capacity" and members["storage"]:
        discharge = dispatch.get("storage_dispatch")
        if discharge is None:
            discharge = dispatch["storage_p"].clip(lower=0.0)
        margin -= discharge.reindex(index=snapshots, columns=members["storage"]).fillna(0.0).sum(axis=1)
    if definition["interconnector_mode"] == "derated_capacity" and members["import_links"]:
        links_p = dispatch["links_p"]
        for name, sign in members["import_links"].items():
            if name in links_p.columns:
                margin -= (sign * links_p[name].reindex(snapshots).fillna(0.0)).clip(lower=0.0)
    return margin


def build_scarcity_pricing_callback(network, curve: dict, logger=None,
                                    excluded_carriers=DEFAULT_EXCLUDED_CARRIERS):
    """PyPSA extra_functionality adding the stepped shortage curve to Stage 1."""
    logger = logger or logging.getLogger(__name__)
    members = margin_members(network, excluded_carriers)
    definition = curve["margin_definition"]
    factors = margin_factors(network, members, definition)
    carriers = network.generators.loc[members["generators"], "carrier"]
    unlisted = sorted(set(carriers[~carriers.isin(list(definition["generator_factors"]))]))
    logger.info(
        "Scarcity pricing: margin over %d GB generators, %d storage units (%s) and %d import links (%s); "
        "requirement %s MW (model margin) from %s; carriers at the default factor %.2f: %s",
        len(members["generators"]), len(members["storage"]), definition["storage_mode"],
        len(members["import_links"]), definition["interconnector_mode"], curve["requirement_mw"],
        curve["source"], definition["default_generator_factor"], ", ".join(unlisted) or "none",
    )

    def scarcity(n, snapshots):
        model = n.model
        snapshot_index = pd.Index(snapshots, name="snapshot")
        terms = []

        # Only members that are LP variables in this window; the de-rated
        # capacity is summed over the same set.
        present = {"generators": [], "storage": [], "import_links": {}}
        gen_p = model.variables["Generator-p"]
        gen_dim = [d for d in gen_p.dims if d != "snapshot"][0]
        present["generators"] = [g for g in members["generators"] if g in gen_p.coords[gen_dim].values]
        if present["generators"]:
            terms.append(-gen_p.sel({gen_dim: present["generators"]}).sum(gen_dim))

        if definition["storage_mode"] == "derated_capacity" and "StorageUnit-p_dispatch" in model.variables:
            dispatch = model.variables["StorageUnit-p_dispatch"]
            su_dim = [d for d in dispatch.dims if d != "snapshot"][0]
            present["storage"] = [s for s in members["storage"] if s in dispatch.coords[su_dim].values]
            if present["storage"]:
                # Discharge is counted in the de-rated capacity, so it is taken
                # off here; charging is not, so it uses margin as demand does.
                terms.append(-dispatch.sel({su_dim: present["storage"]}).sum(su_dim))

        if definition["interconnector_mode"] == "derated_capacity" and "Link-p" in model.variables:
            link_p = model.variables["Link-p"]
            link_dim = [d for d in link_p.dims if d != "snapshot"][0]
            names = [name for name in members["import_links"] if name in link_p.coords[link_dim].values]
            present["import_links"] = {name: members["import_links"][name] for name in names}
            if names:
                # Gross import: g >= sign x p and g >= 0. The LP never wants g
                # larger than it must be (it lowers the margin), so g is the import.
                coords = {"snapshot": snapshot_index, link_dim: pd.Index(names, name=link_dim)}
                gross = model.add_variables(lower=0.0, coords=[coords["snapshot"], coords[link_dim]],
                                            name="Link-scarcity-gross-import")
                sign_da = xr.DataArray(np.array([members["import_links"][k] for k in names], dtype=float),
                                       dims=[link_dim], coords={link_dim: coords[link_dim]})
                model.add_constraints(gross - link_p.sel({link_dim: names}) * sign_da >= 0.0,
                                      name="Link-scarcity-gross-import")
                terms.append(-gross.sum(link_dim))

        capacity = derated_capacity(n, present, definition, snapshots)
        avail_total = xr.DataArray(capacity.to_numpy(dtype=float), dims=["snapshot"],
                                   coords={"snapshot": snapshot_index})

        weights = n.snapshot_weightings.objective.reindex(snapshots).fillna(1.0)
        weight_da = xr.DataArray(weights.to_numpy(dtype=float), dims=["snapshot"],
                                 coords={"snapshot": snapshot_index})
        shortfall_total = None
        objective = None
        for k, (width, price) in enumerate(curve["steps"], start=1):
            upper = np.inf if width is None else xr.DataArray(
                seasonal_series(width, snapshot_index), dims=["snapshot"], coords={"snapshot": snapshot_index})
            step = model.add_variables(lower=0.0, upper=upper, coords=[snapshot_index],
                                       name=f"Scarcity-shortfall-step-{k}")
            shortfall_total = step if shortfall_total is None else shortfall_total + step
            cost = (step * weight_da).sum() * float(price)
            objective = cost if objective is None else objective + cost

        margin = terms[0]
        for term in terms[1:]:
            margin = margin + term
        # margin + shortfall >= requirement, margin = de-rated capacity - delivery
        #   <=>  -sum(delivery) + shortfall >= requirement - de-rated capacity
        requirement = xr.DataArray(requirement_series(curve, snapshot_index), dims=["snapshot"],
                                   coords={"snapshot": snapshot_index})
        model.add_constraints(margin + shortfall_total >= requirement - avail_total,
                              name="Scarcity-margin-requirement")
        model.objective = model.objective + objective

    return scarcity
