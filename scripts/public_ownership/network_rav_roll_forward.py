"""Network RAV roll-forward past the determined price controls (FES 2050 plan, task Y12).

**No official RAV or TNUoS projection exists after 2031.**  RIIO-3 (transmission) ends on
31 March 2031 and NESO's TNUoS view on 2030/31; RIIO-ED3 (distribution) has only a
methodology decision.  For a model year after RIIO-3 the network account therefore needs a
declared roll-forward, and this module is it.  Every input, with its source and status, is
in ``data/market/network_cost_sources/rav_roll_forward_2050.yaml``.

* **Transmission.**  The RIIO-3 period-average RAV (the anchor the 2030 account already uses)
  is rolled forward year by year: programme investment (RIIO-3, then NESO's Beyond 2030
  Update) enters the RAV at the ET capitalisation rate, new additions depreciate straight
  line over 45 years (the RIIO profile), the anchor runs off over a declared remaining life,
  and after the last published plan capex equals depreciation (the network held, decision 10).
* **Distribution.**  The 2030 roll-forward of ED2 is scaled by FES peak demand to a declared
  elasticity (the CCC's method on DESNZ's distribution range); the implied gross additions
  are backed out against the ED2 depreciation run-off and 45-year lives, for the cross-check.

Everything is converted to the model's real 2024 GBP **before** it is added: RIIO-3 figures
are 2023/24 prices, ED2 figures 2020/21 prices, NESO's 2026 figure is taken as 2025 prices,
DESNZ's framework is 2020 prices.  (The rate conversion, RIIO-3 semi-nominal to CPIH-real,
is the account's, :func:`network_finance_account.build_account`.)

Nothing here is read by a 2030 rule.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Mapping

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUTS = PROJECT_ROOT / "data" / "market" / "network_cost_sources" / "rav_roll_forward_2050.yaml"
TRANSMISSION = "transmission"
DISTRIBUTION = "distribution"
AFTER_PLAN_RULES = ("capex_equals_depreciation", "zero_capex")
PATH_COLUMNS = ("year", "opening_rav_gbp", "capex_gbp", "additions_gbp", "depreciation_gbp", "closing_rav_gbp")
NO_PROJECTION_STATEMENT = ("No official RAV or TNUoS projection exists after 2031: RIIO-3 ends 31 March 2031 and "
                           "NESO's TNUoS view ends at 2030/31; RIIO-ED3 has no determination. The asset bases are a "
                           "declared roll-forward.")


def load_inputs(path: str | Path | None = None) -> dict:
    """Read the roll-forward inputs and check that every sourced input says where it is from.

    Args:
        path: The YAML; defaults to ``rav_roll_forward_2050.yaml``.

    Returns:
        The parsed inputs, with ``_path`` set.

    Raises:
        ValueError: A section is missing, or an input lacks its source or status.
    """
    path = Path(path) if path is not None else DEFAULT_INPUTS
    inputs = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section in ("cpih_calendar", TRANSMISSION, DISTRIBUTION):
        if section not in inputs:
            raise ValueError(f"{path}: roll-forward inputs lack the {section!r} section")
    t, d = inputs[TRANSMISSION], inputs[DISTRIBUTION]
    provenance = {
        "cpih_calendar": inputs["cpih_calendar"],
        "transmission.anchor_remaining_life_years": t["anchor_remaining_life_years"],
        "transmission.depreciation_life_years": t["depreciation_life_years"],
        "transmission.capitalisation_rate": t["capitalisation_rate"],
        "transmission.programmes.riio3": t["programmes"]["riio3"],
        "transmission.programmes.beyond_2030_update": t["programmes"]["beyond_2030_update"],
        "transmission.programmes.after_plan": t["programmes"]["after_plan"],
        "distribution.peak_elasticity": d["peak_elasticity"],
        "distribution.peak_demand_gw": d["peak_demand_gw"],
        "distribution.anchor_depreciation": d["anchor_depreciation"],
        "distribution.depreciation_life_years": d["depreciation_life_years"],
    }
    for name, entry in provenance.items():
        if not (isinstance(entry, Mapping) and entry.get("source") and entry.get("status")):
            raise ValueError(f"{path}: {name} needs a source and a status (provenance on every input)")
    if t["programmes"]["after_plan"]["rule"] not in AFTER_PLAN_RULES:
        raise ValueError(f"{path}: transmission after_plan rule must be one of {AFTER_PLAN_RULES}")
    if d.get("rule") != "peak_elasticity":
        raise ValueError(f"{path}: distribution rule {d.get('rule')!r}; supported: peak_elasticity")
    inputs["_path"] = str(path)
    return inputs


def rebase_factor(basis: str, inputs: Mapping, fy_price_factor: Callable[[str], float] | None = None) -> float:
    """Multiplier from a published price basis to the model's real 2024 GBP (ONS CPIH).

    Args:
        basis: A calendar year ("2025") from ``cpih_calendar``, or a financial year
            ("2023/24") resolved by ``fy_price_factor``.
        inputs: The roll-forward inputs.
        fy_price_factor: Financial-year factor; defaults to
            :func:`network_finance_account.price_factor`, the 2030 account's own.

    Returns:
        The factor.
    """
    basis = str(basis)
    if "/" in basis:
        if fy_price_factor is None:
            from scripts.public_ownership.network_finance_account import price_factor as fy_price_factor
        return float(fy_price_factor(basis))
    values = {str(k): float(v) for k, v in inputs["cpih_calendar"]["values"].items()}
    target = str(inputs.get("model_price_year", 2024))
    if basis not in values or target not in values:
        raise ValueError(f"no CPIH calendar average for {basis!r} or {target!r}; declared: {sorted(values)}")
    return values[target] / values[basis]


def _spread(total_gbp: float, years: tuple[int, int]) -> dict[int, float]:
    first, last = int(years[0]), int(years[1])
    n = last - first + 1
    return {y: total_gbp / n for y in range(first, last + 1)}


def transmission_programme(inputs: Mapping, *, sensitivity: str | None = None,
                           fy_price_factor: Callable[[str], float] | None = None) -> dict:
    """Programme capex by calendar year (real 2024 GBP) and the top-up the anchor lacks.

    Args:
        inputs: The roll-forward inputs.
        sensitivity: None (central) or a key of ``transmission.sensitivities``.
        fy_price_factor: See :func:`rebase_factor`.

    Returns:
        ``{"capex_by_year": {year: gbp}, "anchor_top_up_capex_gbp": gbp, "capitalisation_rate": x, ...}``.
    """
    t = inputs[TRANSMISSION]
    sens = dict((t.get("sensitivities") or {}).get(sensitivity) or {}) if sensitivity else {}
    if sensitivity and not sens:
        raise ValueError(f"unknown transmission sensitivity {sensitivity!r}")
    riio3 = dict(t["programmes"]["riio3"])
    total = float(sens.get("riio3_value_gbp_bn", riio3["value_gbp_bn"]))
    committed = min(float(riio3["committed_gbp_bn"]), total)
    f3 = rebase_factor(riio3["price_basis"], inputs, fy_price_factor)
    first, last = (int(v) for v in riio3["years"])
    anchor = int(t["anchor_year"])
    per_year = total * 1e9 * f3 / (last - first + 1)
    capex = {y: per_year for y in range(anchor + 1, last + 1)}
    pre_anchor_share = (anchor - first + 1) / (last - first + 1)
    top_up = (total - committed) * 1e9 * f3 * pre_anchor_share
    b2030 = {**dict(t["programmes"]["beyond_2030_update"]), **dict(sens.get("beyond_2030_update") or {})}
    f_b = rebase_factor(b2030["price_basis"], inputs, fy_price_factor)
    for year, gbp in _spread(float(b2030["value_gbp_bn"]) * 1e9 * f_b, tuple(b2030["years"])).items():
        capex[year] = capex.get(year, 0.0) + gbp
    cap = float(sens.get("capitalisation_rate", t["capitalisation_rate"]["value"]))
    return {"capex_by_year": capex, "anchor_top_up_capex_gbp": top_up, "capitalisation_rate": cap,
            "riio3_gbp_bn": total, "riio3_committed_gbp_bn": committed, "riio3_rebase_factor": f3,
            "beyond_2030_gbp_bn": float(b2030["value_gbp_bn"]), "beyond_2030_price_basis": str(b2030["price_basis"]),
            "beyond_2030_rebase_factor": f_b, "beyond_2030_years": list(b2030["years"]),
            "after_plan": dict(t["programmes"]["after_plan"])}


def _roll(anchor_year: int, anchor_rav: float, end_year: int, *, life: float,
          anchor_run_off: float, capex_for: Callable[[int, float], float] | None = None,
          target_for: Callable[[int], float] | None = None, capitalisation: float = 1.0,
          top_up_additions: float = 0.0) -> pd.DataFrame:
    """Year-end balances: opening + additions - depreciation, vintage by vintage.

    The anchor runs off by ``anchor_run_off`` a year until exhausted; every later addition
    (and ``top_up_additions`` at the anchor) depreciates straight line over ``life`` from
    the year after it is added.  Either ``capex_for(year, depreciation)`` sets capex (RAV
    additions = capitalisation x capex), or ``target_for(year)`` sets the closing RAV and
    the additions are backed out (never negative).
    """
    vintages: list[tuple[int, float]] = [(anchor_year, float(top_up_additions))] if top_up_additions else []
    anchor_left = float(anchor_rav)
    closing = float(anchor_rav) + float(top_up_additions)
    rows = [{"year": anchor_year, "opening_rav_gbp": float(anchor_rav), "capex_gbp": top_up_additions / capitalisation
             if capitalisation else 0.0, "additions_gbp": float(top_up_additions), "depreciation_gbp": 0.0,
             "closing_rav_gbp": closing}]
    for year in range(anchor_year + 1, int(end_year) + 1):
        opening = closing
        run_off = min(anchor_left, float(anchor_run_off))
        anchor_left -= run_off
        dep = run_off + sum(amount / life for v, amount in vintages if v < year <= v + life)
        if target_for is not None:
            additions = max(float(target_for(year)) - opening + dep, 0.0)
            capex = additions / capitalisation if capitalisation else 0.0
        else:
            capex = float(capex_for(year, dep))
            additions = capitalisation * capex
        closing = opening + additions - dep
        vintages.append((year, additions))
        rows.append({"year": year, "opening_rav_gbp": opening, "capex_gbp": capex, "additions_gbp": additions,
                     "depreciation_gbp": dep, "closing_rav_gbp": closing})
    return pd.DataFrame(rows, columns=list(PATH_COLUMNS))


def transmission_path(anchor_rav_gbp: float, inputs: Mapping, end_year: int, *, sensitivity: str | None = None,
                      fy_price_factor: Callable[[str], float] | None = None) -> tuple[pd.DataFrame, dict]:
    """The transmission RAV, year by year from the RIIO-3 anchor to ``end_year`` (real 2024 GBP).

    Args:
        anchor_rav_gbp: The RIIO-3 period-average RAV already on the model basis.
        inputs: The roll-forward inputs.
        end_year: The last calendar year.
        sensitivity: None (central) or a key of ``transmission.sensitivities``.
        fy_price_factor: See :func:`rebase_factor`.

    Returns:
        (path, programme) where ``path`` has :data:`PATH_COLUMNS`.
    """
    t = inputs[TRANSMISSION]
    programme = transmission_programme(inputs, sensitivity=sensitivity, fy_price_factor=fy_price_factor)
    cap = programme["capitalisation_rate"]
    after = programme["after_plan"]
    after_first = int(after["years"][0])

    def capex_for(year: int, dep: float) -> float:
        if year >= after_first:
            return dep / cap if after["rule"] == "capex_equals_depreciation" and cap else 0.0
        return programme["capex_by_year"].get(year, 0.0)

    path = _roll(int(t["anchor_year"]), float(anchor_rav_gbp), end_year,
                 life=float(t["depreciation_life_years"]["value"]),
                 anchor_run_off=float(anchor_rav_gbp) / float(t["anchor_remaining_life_years"]["value"]),
                 capex_for=capex_for, capitalisation=cap,
                 top_up_additions=cap * programme["anchor_top_up_capex_gbp"])
    return path, programme


def peak_gw(year: int, inputs: Mapping) -> float:
    """FES peak demand in a year, geometric between the declared points (held outside them)."""
    points = sorted((int(k), float(v)) for k, v in inputs[DISTRIBUTION]["peak_demand_gw"]["values"].items())
    if year <= points[0][0]:
        return points[0][1]
    for (y0, p0), (y1, p1) in zip(points, points[1:]):
        if y0 <= year <= y1:
            return p0 * (p1 / p0) ** ((year - y0) / (y1 - y0))
    return points[-1][1]


def distribution_path(anchor_rav_gbp: float, anchor_depreciation_gbp: float, inputs: Mapping,
                      end_year: int) -> pd.DataFrame:
    """The distribution RAV from the 2030 roll-forward, scaled by FES peak demand (real 2024 GBP).

    Args:
        anchor_rav_gbp: The 2030 account's distribution RAV on the model basis.
        anchor_depreciation_gbp: The ED2 final-year depreciation on the model basis.
        inputs: The roll-forward inputs.
        end_year: The last calendar year.

    Returns:
        The path, with :data:`PATH_COLUMNS` (capex = RAV additions: capitalisation 1).
    """
    d = inputs[DISTRIBUTION]
    anchor = int(d["anchor_year"])
    elasticity = float(d["peak_elasticity"]["value"])
    base_peak = peak_gw(anchor, inputs)

    def target_for(year: int) -> float:
        return float(anchor_rav_gbp) * (peak_gw(year, inputs) / base_peak) ** elasticity

    return _roll(anchor, float(anchor_rav_gbp), end_year, life=float(d["depreciation_life_years"]["value"]),
                 anchor_run_off=float(anchor_depreciation_gbp), target_for=target_for)


def rav_at(path: pd.DataFrame, year: int) -> float:
    """The year-end RAV of ``year`` on a path."""
    row = path.loc[path["year"].eq(int(year))]
    if row.empty:
        raise ValueError(f"the roll-forward path does not reach {year} (it spans {int(path['year'].min())} to "
                         f"{int(path['year'].max())})")
    return float(row["closing_rav_gbp"].iloc[0])


def capex_between(path: pd.DataFrame, first: int, last: int) -> float:
    """Capex summed over calendar years ``first`` to ``last`` inclusive."""
    rows = path.loc[path["year"].between(int(first), int(last))]
    return float(rows["capex_gbp"].sum())


def roll_forward(blocks_2024: Mapping, inputs: Mapping, model_year: int, *,
                 fy_price_factor: Callable[[str], float] | None = None) -> dict:
    """Both RAVs in ``model_year``, their paths, the cross-checks and the sensitivities.

    Args:
        blocks_2024: ``{"transmission_anchor_rav_gbp", "distribution_anchor_rav_gbp",
            "distribution_anchor_depreciation_gbp"}``, all already on the model basis.
        inputs: The roll-forward inputs.
        model_year: The calendar model year.
        fy_price_factor: See :func:`rebase_factor`.

    Returns:
        ``transmission_rav_<year>_gbp``, ``distribution_rav_<year>_gbp``, the paths (records),
        the programme, the cross-checks against the workbook and DESNZ, and the transmission
        sensitivities.
    """
    year = int(model_year)
    t_path, programme = transmission_path(blocks_2024["transmission_anchor_rav_gbp"], inputs, year,
                                          fy_price_factor=fy_price_factor)
    d_path = distribution_path(blocks_2024["distribution_anchor_rav_gbp"],
                               blocks_2024["distribution_anchor_depreciation_gbp"], inputs, year)
    t_rav, d_rav = rav_at(t_path, year), rav_at(d_path, year)
    checks = cross_checks(t_path, d_path, inputs, fy_price_factor=fy_price_factor)
    sensitivities = {}
    for name, sens in dict(inputs[TRANSMISSION].get("sensitivities") or {}).items():
        if name.startswith("cross_check"):
            continue
        path, _ = transmission_path(blocks_2024["transmission_anchor_rav_gbp"], inputs, year, sensitivity=name,
                                    fy_price_factor=fy_price_factor)
        sensitivities[name] = {"note": sens.get("note", ""), f"transmission_rav_{year}_gbp": rav_at(path, year),
                               "capex_2031_to_2050_gbp": capex_between(path, 2031, 2050)}
    d = inputs[DISTRIBUTION]
    return {
        "statement": NO_PROJECTION_STATEMENT,
        "inputs_file": inputs.get("_path"),
        "model_year": year,
        f"transmission_rav_{year}_gbp": t_rav,
        f"distribution_rav_{year}_gbp": d_rav,
        "transmission_basis": (
            f"declared roll-forward: RIIO-3 period-average RAV at end-{inputs[TRANSMISSION]['anchor_year']}, "
            f"RIIO-3 at {programme['riio3_gbp_bn']:g} bn (2023/24) and the Beyond 2030 Update at "
            f"{programme['beyond_2030_gbp_bn']:g} bn ({programme['beyond_2030_price_basis']} prices, assumed) "
            f"over {programme['beyond_2030_years'][0]}-{programme['beyond_2030_years'][1]}, capitalised at "
            f"{programme['capitalisation_rate']:.0%}, 45-year straight-line depreciation, capex = depreciation "
            f"from {programme['after_plan']['years'][0]}; no official RAV projection exists after 2031"),
        "distribution_basis": (
            f"declared roll-forward: the 2030 ED2 roll-forward scaled by FES HT peak demand "
            f"({peak_gw(int(d['anchor_year']), inputs):.2f} GW to {peak_gw(year, inputs):.2f} GW) at elasticity "
            f"{float(d['peak_elasticity']['value']):g} (the CCC's method on DESNZ's range); ED3 has no "
            "determination and no official RAV projection exists after 2031"),
        "transmission_path": t_path.to_dict(orient="records"),
        "distribution_path": d_path.to_dict(orient="records"),
        "transmission_programme": {k: v for k, v in programme.items() if k != "capex_by_year"},
        "cross_checks": checks,
        "transmission_sensitivities": sensitivities,
    }


def cross_checks(t_path: pd.DataFrame, d_path: pd.DataFrame, inputs: Mapping, *,
                 fy_price_factor: Callable[[str], float] | None = None) -> dict:
    """The roll-forward's 2030-2050 capex against the workbook's GBP 160 bn and DESNZ's ranges.

    Args:
        t_path: The transmission path.
        d_path: The distribution path.
        inputs: The roll-forward inputs.
        fy_price_factor: See :func:`rebase_factor`.

    Returns:
        The comparisons, in real 2024 GBP bn.
    """
    out: dict = {}
    wb = (inputs.get("cross_checks") or {}).get("workbook_network_capex")
    last = int(min(t_path["year"].max(), d_path["year"].max()))
    t_2031 = capex_between(t_path, 2031, last)
    t_2030 = capex_between(t_path, 2030, last)
    if wb:
        out["workbook_network_capex"] = {
            "workbook_gbp_bn": float(wb["value_gbp_bn"]), "workbook_basis": wb["basis"],
            "roll_forward_transmission_capex_2031_to_end_gbp_bn": t_2031 / 1e9,
            "roll_forward_transmission_capex_2030_to_end_gbp_bn": t_2030 / 1e9,
            "ratio_2031_to_end": t_2031 / 1e9 / float(wb["value_gbp_bn"]),
            "roll_forward_distribution_additions_2031_to_end_gbp_bn": capex_between(d_path, 2031, last) / 1e9,
            "note": ("the workbook's 160 bn is transmission ('inv_tx_50'); the roll-forward's transmission capex "
                     "is the comparable figure, distribution additions are shown beside it"),
        }
    ensf = (inputs[TRANSMISSION].get("sensitivities") or {}).get("cross_check_desnz_ensf")
    if ensf:
        f = rebase_factor(ensf["price_basis"], inputs, fy_price_factor)
        out["desnz_ensf_transmission"] = {"range_gbp_bn_2024": [v * f for v in ensf["range_gbp_bn"]],
                                          "roll_forward_capex_2031_to_end_gbp_bn": t_2031 / 1e9,
                                          "note": ensf.get("note", "")}
    dist = inputs[DISTRIBUTION].get("cross_checks") or {}
    if dist.get("desnz_ensf_distribution"):
        e = dist["desnz_ensf_distribution"]
        f = rebase_factor(e["price_basis"], inputs, fy_price_factor)
        out["desnz_ensf_distribution"] = {"range_gbp_bn_2024": [v * f for v in e["range_gbp_bn"]],
                                          "roll_forward_gross_additions_2031_to_end_gbp_bn":
                                              capex_between(d_path, 2031, last) / 1e9,
                                          "source": e["source"],
                                          "note": "DESNZ's range is load-related only, and 2021-2050; the roll-forward "
                                                  "adds replacement as well"}
    if dist.get("ofgem_ed_rav_growth"):
        first, end = d_path.iloc[0], d_path.iloc[-1]
        years = int(end["year"]) - int(first["year"])
        growth = (float(end["closing_rav_gbp"]) / float(first["closing_rav_gbp"])) ** (1.0 / years) - 1.0 if years else 0.0
        out["ofgem_ed_rav_growth"] = {"ofgem_assumption": float(dist["ofgem_ed_rav_growth"]["value"]),
                                      "roll_forward_real_growth": growth,
                                      "source": dist["ofgem_ed_rav_growth"]["source"]}
    return out
