"""Household archetypes with a standing and a unit split (item G6).

Decisions 11 and 12 (22 September 2026): Ofgem's low, medium and high typical
domestic consumption values plus a heat-pump and an electric-vehicle household,
reported as points. A single-rate tariff charges every household the same unit rate,
so an archetype pays

    bill(kWh) = standing charge + unit rate x kWh,

where each line's standing part is ``standing share x its value at the benchmark``
and the rest is recovered per kWh. The 3,100 kWh benchmark bill is unchanged by
construction.

**The shares.** Each line's share is declared in the assumptions file
(``charge_basis``) and defaults to zero, a unit-rate line. TNUoS takes the residual
share of its own derivation. Supplier costs, DUoS and the smart-meter allowance are
the mixed lines the Ofgem cap model splits between nil and benchmark consumption; they
share one standing share, calibrated as the historical path does
(``scripts/market/build_household_bill_component_register.py``) so the known fixed
items plus the mixed lines reproduce Ofgem's published nil-consumption charge. A
public-ownership saving on a line inherits that line's share, so a network saving
lands where the network charge does.

**Composed households (task Y13, decisions 12 and 12a).** An archetype's ``added`` is
either one set of inputs (the 2030 form) or a list of parts, each
``{"part": name, "inputs": {...}}``, so a heat-pump plus EV household is composed as
base + heat-pump addition + EV addition in one archetype. Two rules take a declared
efficiency gain on the 2030 inputs (FES 2025 figure F.05 publishes gains, not levels),
and ``annual_kwh`` adds a published per-unit average directly. With
``report_composition: true`` the audit lists every archetype's parts and inputs.
"""

from typing import Mapping

import pandas as pd

VALUE = "annual_gbp_per_3100kwh_ex_vat"
ADDITIONS = {
    # heat pump: the gas a medium household burns, as heat, over the pump's seasonal efficiency
    frozenset({"gas_kwh", "boiler_efficiency", "seasonal_performance_factor"}):
        lambda a: a["gas_kwh"] * a["boiler_efficiency"] / a["seasonal_performance_factor"],
    # electric vehicle: annual miles at the wall-plug energy per mile
    frozenset({"miles", "kwh_per_mile"}): lambda a: a["miles"] * a["kwh_per_mile"],
    # heat pump with a seasonal-efficiency gain on the SPF (Y13: FES F.05 SCOP improvement)
    frozenset({"gas_kwh", "boiler_efficiency", "seasonal_performance_factor", "seasonal_performance_gain"}):
        lambda a: a["gas_kwh"] * a["boiler_efficiency"]
        / (a["seasonal_performance_factor"] * (1.0 + a["seasonal_performance_gain"])),
    # electric vehicle with an efficiency gain on the energy per mile (Y13: FES F.05 EV efficiency)
    frozenset({"miles", "kwh_per_mile", "efficiency_gain"}):
        lambda a: a["miles"] * a["kwh_per_mile"] / (1.0 + a["efficiency_gain"]),
    # a published per-unit annual electricity use (Y13: FES per-unit averages)
    frozenset({"annual_kwh"}): lambda a: a["annual_kwh"],
}


def _addition_kwh(name: str, inputs: Mapping) -> float:
    """The kWh one set of addition inputs adds, by the rule its input names select."""
    added = {k: float(v) for k, v in inputs.items()}
    rule = ADDITIONS.get(frozenset(added))
    if rule is None:
        raise ValueError(f"archetype {name!r}: added inputs {sorted(added)} match no rule; "
                         f"expected one of {[sorted(k) for k in ADDITIONS]}")
    return float(rule(added))


def _parts(spec: Mapping) -> list[tuple[str, Mapping]]:
    """The declared additions of one archetype as ``(part, inputs)``, in order."""
    added = spec.get("added") or {}
    if isinstance(added, Mapping):
        return [(str(spec.get("part", "added")), added)]
    parts = []
    for item in added:
        if "part" not in item or "inputs" not in item:
            raise ValueError(f"archetype {spec['id']!r}: each listed addition needs a part and its inputs")
        parts.append((str(item["part"]), item["inputs"]))
    if len({p for p, _ in parts}) != len(parts):
        raise ValueError(f"archetype {spec['id']!r}: a part is named twice")
    return parts


def archetype_composition(block: Mapping) -> dict[str, list[dict]]:
    """Each archetype as its parts: the base's own parts, then each addition with its inputs and kWh.

    Args:
        block: the assumptions file's ``household_archetypes``.

    Returns:
        ``{archetype id: [{"part", "kwh", "from" or "inputs"}, ...]}``; the parts sum to the
        archetype's :func:`archetype_consumption`.
    """
    out: dict[str, list[dict]] = {}
    for spec in block["archetypes"]:
        name = str(spec["id"])
        if "kwh" in spec:
            out[name] = [{"part": "base", "from": name, "kwh": float(spec["kwh"])}]
            continue
        base = spec.get("base")
        if base not in out:
            raise ValueError(f"archetype {name!r}: base {base!r} must be an archetype declared before it")
        parts = [dict(p) for p in out[base]]
        for part, inputs in _parts(spec):
            parts.append({"part": part, "kwh": _addition_kwh(name, inputs),
                          "inputs": {k: float(v) for k, v in inputs.items()}})
        out[name] = parts
    return out


def archetype_consumption(block: Mapping) -> dict[str, float]:
    """Annual kWh per archetype, in the declared order."""
    out: dict[str, float] = {}
    for spec in block["archetypes"]:
        name = str(spec["id"])
        if "kwh" in spec:
            out[name] = float(spec["kwh"])
            continue
        base = spec.get("base")
        if base not in out:
            raise ValueError(f"archetype {name!r}: base {base!r} must be an archetype declared before it")
        total = out[base]
        for _, inputs in _parts(spec):
            total += _addition_kwh(name, inputs)
        out[name] = total
    return out


def mixed_standing_share(calibration: Mapping) -> float:
    """(published nil charge - known fixed items) / the mixed lines, on the cap's own basis."""
    nil = float(calibration["nil_gbp_ex_vat"])
    fixed = sum(float(v) for v in calibration["known_fixed_gbp"].values())
    mixed = sum(float(v) for v in calibration["mixed_lines_gbp"].values())
    share = (nil - fixed) / mixed if mixed > 0 else float("nan")
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"calibrated standing share {share:.3f} is not between 0 and 1; check the calibration inputs")
    return share


def standing_shares(components: pd.DataFrame, basis: Mapping) -> pd.Series:
    """Standing share per component id: declared, inherited, or zero (a unit-rate line)."""
    declared = {str(k): float(v) for k, v in (basis.get("standing_share") or {}).items()}
    for key, value in declared.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"standing share for {key!r} is {value}, not between 0 and 1")
    inherit = {str(k): str(v) for k, v in (basis.get("inherit") or {}).items()}
    ids = components["component_id"].astype(str)
    return pd.Series([declared.get(i, declared.get(inherit.get(i, ""), 0.0)) for i in ids],
                     index=ids.to_numpy(), dtype=float)


def archetype_bills(components: pd.DataFrame, shares: pd.Series, consumption: Mapping[str, float], *,
                    benchmark_kwh: float, vat_rate: float) -> pd.DataFrame:
    """One row per archetype: standing charge, unit rate, bill ex VAT, VAT (once) and total."""
    values = components.set_index(components["component_id"].astype(str))[VALUE].astype(float)
    shares = shares.reindex(values.index).fillna(0.0)
    standing = float((values * shares).sum())
    unit_at_benchmark = float((values * (1.0 - shares)).sum())
    rows = []
    for name, kwh in consumption.items():
        ex_vat = standing + unit_at_benchmark * float(kwh) / float(benchmark_kwh)
        rows.append({
            "archetype": name, "annual_consumption_kwh": float(kwh),
            "standing_charge_gbp_ex_vat": standing,
            "unit_rate_p_per_kwh_ex_vat": unit_at_benchmark / float(benchmark_kwh) * 100.0,
            "ex_vat_gbp": ex_vat, "vat_gbp": ex_vat * float(vat_rate), "total_gbp_inc_vat": ex_vat * (1.0 + float(vat_rate)),
        })
    return pd.DataFrame(rows).set_index("archetype")


def resolve_charge_basis(components: pd.DataFrame, assumptions: Mapping, root=".") -> tuple[dict, dict]:
    """The declared shares for this bill, with the mixed share calibrated; and an audit.

    TNUoS takes its residual share from its own derivation. The mixed lines share one
    standing share, calibrated so the known fixed items plus the mixed lines at their
    values in this bill reproduce the published nil-consumption charge.
    """
    from pathlib import Path

    cfg = assumptions["charge_basis"]
    sourced = assumptions.get("sourced_externals") or {}
    shares: dict[str, float] = {str(k): float(v) for k, v in (cfg.get("standing_share") or {}).items()}
    if "calibration" not in cfg:
        # A basis declared outright (tests, sensitivities): no calibration to run.
        return {"standing_share": shares, "inherit": dict(cfg.get("inherit") or {})}, {"standing_share": dict(shares)}
    for key, spec in (cfg.get("from_derivation") or {}).items():
        derivation = sourced[key]["derivation"]
        shares[key] = float(derivation[spec["numerator"]]) / float(derivation[spec["denominator"]])
    values = components.set_index(components["component_id"].astype(str))[VALUE].astype(float)
    cal = cfg["calibration"]
    tariffs = pd.read_csv(Path(root) / cal["nil_source_file"])
    tariffs = tariffs.loc[tariffs["payment_method"].astype(str).eq(str(cal["payment_method"]))]
    if tariffs.empty:
        raise ValueError(f"{cal['nil_source_file']} has no {cal['payment_method']} rows")
    known = {}
    for name, spec in cal["known_fixed"].items():
        known[name] = float(spec["gbp_per_site_day"]) * 365.0 if "gbp_per_site_day" in spec else float(spec["gbp"])
    missing = [line for line in cfg["mixed_lines"] if line not in values.index]
    if missing:
        raise ValueError(f"charge_basis mixed lines {missing} are not in the bill")
    uplift = float(cal["mixed_lines_uplift_to_2026"])
    calibration = {
        "nil_gbp_ex_vat": float(tariffs["nil_gbp_per_year_ex_vat"].mean()),
        "known_fixed_gbp": known,
        "mixed_lines_gbp": {line: float(values[line]) * uplift for line in cfg["mixed_lines"]},
    }
    mixed = mixed_standing_share(calibration)
    for line in cfg["mixed_lines"]:
        shares[line] = mixed
    audit = {**calibration, "mixed_standing_share": mixed, "standing_share": dict(shares),
             "nil_source_file": cal["nil_source_file"], "payment_method": cal["payment_method"]}
    return {"standing_share": shares, "inherit": dict(cfg.get("inherit") or {})}, audit


def project_archetypes(components: pd.DataFrame, assumptions: Mapping, *, vat_rate: float, root=".") -> dict:
    """The benchmark, the alternative volumes and every archetype on one charge basis."""
    block = assumptions["household_archetypes"]
    benchmark_kwh = float(block["benchmark_kwh"])
    basis, audit = resolve_charge_basis(components, assumptions, root=root)
    shares = standing_shares(components, basis)
    volumes = {"benchmark": benchmark_kwh, **archetype_consumption(block)}
    for kwh in assumptions["benchmark"].get("alternative_consumption_kwh") or []:
        volumes.setdefault(f"alternative_{int(kwh)}", float(kwh))
    bills = archetype_bills(components, shares, volumes, benchmark_kwh=benchmark_kwh, vat_rate=vat_rate)
    out = {"bills": bills.round(6).reset_index().to_dict(orient="records"),
           "charge_basis": audit,
           "domestic_customer_accounts": block.get("domestic_customer_accounts"),
           "standing_revenue_cross_check_gbp": (
               float(bills.loc["benchmark", "standing_charge_gbp_ex_vat"])
               * float((block.get("domestic_customer_accounts") or {}).get("value", float("nan"))))}
    # Task Y13: the composed households' parts, for the cost-to-serve diagnostic and the audit.
    # Only when the file asks, so a bill that does not (the 2030 year) is unchanged.
    if block.get("report_composition"):
        out["composition"] = archetype_composition(block)
    # 13 Nov 2026: the headline archetype whenever the file names one (the public-ownership layer reports it beside
    # the benchmark, scripts/public_ownership/electrified_household.py).
    if block.get("headline_archetype"):
        if str(block["headline_archetype"]) not in volumes:
            raise ValueError(f"headline archetype {block['headline_archetype']!r} is not an archetype of this bill")
        out["headline_archetype"] = str(block["headline_archetype"])
    # 13 Nov 2026 (user): a high-demand heat-pump plus EV household beside the headline (2040 and 2050).
    if block.get("high_demand_archetype"):
        if str(block["high_demand_archetype"]) not in volumes:
            raise ValueError(f"high-demand archetype {block['high_demand_archetype']!r} is not an archetype of this bill")
        out["high_demand_archetype"] = str(block["high_demand_archetype"])
    return out


#: Ofgem's 2026 typical domestic consumption values (TDCV decision, May 2026, Table 1; from 1 July 2026).
OFGEM_2026_HIGH_GAS_KWH = 14000
OFGEM_TDCV_SOURCE = "Ofgem TDCV decision, May 2026, Table 1 (data/market/consumer_bill_sources/archetypes/ofgem_tdcv_review_decision_2026_05.pdf)"


def add_today_base_households(block: dict, archetypes_2030: Mapping, composed_id: str) -> None:
    """User decision (13 Nov 2026): the headline electrified household on today's household in every year.

    Adds Ofgem's 2026 medium and high households (the 2030 file's ``ofgem_medium`` and ``ofgem_high``) and two
    heat-pump plus EV households composed on them with ``composed_id``'s parts (the year's heat-pump and EV
    efficiency gains): ``<composed_id>_today_base`` (the headline, so 2030, 2040 and 2050 share one base) and
    ``<composed_id>_high_demand`` (Ofgem's high household, its heat pump sized on Ofgem's high gas use). The
    FES-based households stay in the file beside them. ``block`` is changed in place.
    """
    import copy

    by_id = {str(a["id"]): a for a in block["archetypes"]}
    base_2030 = {str(a["id"]): a for a in archetypes_2030["archetypes"]}
    composed = by_id[composed_id]
    for name in ("ofgem_medium", "ofgem_high"):
        if name not in by_id:
            block["archetypes"].append(copy.deepcopy(base_2030[name]))
    today = {"id": f"{composed_id}_today_base", "base": "ofgem_medium", "added": copy.deepcopy(composed["added"]),
             "source": ("User, 13 Nov 2026: the headline electrified household on today's household in every year "
                        "(Ofgem's 2026 medium, 2,500 kWh, as the 2030 heat_pump_ev_2030), with this year's heat-pump "
                        f"and EV parts ({composed_id}); the FES-based household is kept beside it")}
    high_added = copy.deepcopy(composed["added"])
    for part in high_added:
        if part["part"] == "heat_pump":
            part["inputs"]["gas_kwh"] = OFGEM_2026_HIGH_GAS_KWH
            part["source"] = (f"As {composed_id}'s heat pump, sized on Ofgem's 2026 high gas TDCV "
                              f"({OFGEM_2026_HIGH_GAS_KWH:,} kWh; {OFGEM_TDCV_SOURCE})")
    high = {"id": f"{composed_id}_high_demand", "base": "ofgem_high", "added": high_added,
            "source": ("User, 13 Nov 2026: a high-demand heat-pump plus EV household beside the headline: Ofgem's 2026 "
                       "high household (3,800 kWh) with a heat pump sized on Ofgem's high gas use and the same car")}
    block["archetypes"].extend([today, high])
    block["headline_archetype"] = today["id"]
    block["high_demand_archetype"] = high["id"]
    block["purpose"] = (str(block.get("purpose", "")) + " User, 13 Nov 2026: the headline is the same household on "
                        "today's base (Ofgem medium) in every year, with a high-demand household beside it; the FES-based "
                        "households are kept.")
