"""Generate the consumer-bill methodology chapter's run figures from the bill outputs.

The bill chapter (``docs/source/model/after_the_solve/consumer_bill/``) works through the
CP30 2030 bill with figures from one run. This script writes them, so the text
cannot drift from the outputs:

* one MyST table per worked-example table, ``_generated/<name>.md``, which the
  pages pull in with ``{include}``;
* ``_generated/values.json``: named scalars (``{"values": {name: text}}``) that
  ``docs/source/conf.py`` loads as ``myst_substitutions``, used in the prose as
  ``{{ name }}``.

Every table starts with a provenance line (the bill run's date and its input
files). Inputs are the bill outputs and the files the bill read; each is a CLI
argument. A missing input is an error, and so is an input that disagrees with
the bill summary (a stale output): the script never writes tables from a mix of
runs. Published figures (Ofgem rates, auction prices, OBR forecasts) stay as
text in the pages unless they arrive in an input the script reads anyway.

Run:  python scripts/analysis/build_bill_docs_values.py   (rule ``build_bill_docs_values``)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.consumer_bill import bill_groupings  # noqa: E402
from scripts.market.cfd_strike_table import MODELLED_ROUND  # noqa: E402

RESOURCES = ROOT / "resources"
OUT_DIR = ROOT / "docs" / "source" / "model" / "after_the_solve" / "consumer_bill" / "_generated"
HOUSEHOLD_MWH = 3.1
VALUE = bill_groupings.VALUE

DEFAULT_INPUTS = {
    "components": RESOURCES / "analysis" / "CP30_2030_cp30_network_consumer_bill_future_components.csv",
    "summary": RESOURCES / "analysis" / "CP30_2030_cp30_network_consumer_bill_future_summary.json",
    "hourly": RESOURCES / "analysis" / "CP30_2030_cp30_network_consumer_bill_future_hourly_inputs.csv",
    "ro_summary": RESOURCES / "analysis" / "CP30_2030_cp30_network_ro_scheme_summary.json",
    "ro_fleet": RESOURCES / "analysis" / "CP30_2030_cp30_network_ro_eligible_fleet.csv",
    "balancing": RESOURCES / "analysis" / "CP30_2030_cp30_network_future_balancing_costs_summary.json",
    "cm_calendar": RESOURCES / "market" / "capacity_market" / "cm_calendar_2030_summary.json",
    "cm_auction": RESOURCES / "market" / "capacity_market" / "cm_auction_model.json",
    "public_account": RESOURCES / "public_ownership" / "cp30_2030_cp30_network_public_account_summary.json",
    "recovery": ROOT / "data" / "market" / "consumer_bill_forecast" / "2030_recovery_demand_assumptions.json",
    "line_sources": ROOT / "data" / "market" / "consumer_bill_benchmarks" / "bill_line_sources.csv",
    "groupings": bill_groupings.GROUPINGS_PATH,
    "grouping_views": bill_groupings.VIEWS_PATH,
}
#: Read when present (the 2030/31 T-4 supply by unit, written since the CP30-fleet auction).
OPTIONAL_INPUTS = {
    "cm_supply": RESOURCES / "market" / "capacity_market" / "cm_t4_2030_31_supply.csv",
}

#: The generated tables, in the order the chapter uses them (file stem -> page).
TABLES = {
    "bill_lines": "index",
    "group_totals": "index",
    "source_kinds": "index",
    "wholesale_quarters": "wholesale",
    "wholesale_build_up": "wholesale",
    "wholesale_sensitivities": "wholesale",
    "cfd_by_technology": "cfd",
    "cfd_largest_contracts": "cfd",
    "cm_delivery_years": "capacity_market",
    "cm_supply": "capacity_market",
    "cm_cases": "capacity_market",
    "ro_by_carrier": "policy_levies",
    "network_constraint_build_up": "networks",
    "recovery_volumes": "recovery",
    "recovery_cfd_example": "recovery",
    "supplier_components": "supplier",
}
VALUES_FILE = "values.json"
GENERATED_FILES = [f"{name}.md" for name in TABLES] + [VALUES_FILE]

# ── Labels the chapter uses ──────────────────────────────────────────────────
LINE_LABELS = {
    "direct_energy": "Wholesale energy",
    "cfd": "Contracts for Difference",
    "capacity_market": "Capacity Market",
    "procurement_losses": "Losses",
    "shaping_adjustment": "Shaping",
    "remaining_ccs": "Net Zero Teesside Power DPA",
    "allocated_ccs": "Power BECCS CfD, electricity",
    "procurement_imbalance": "Imbalance",
    "procurement_additional_risk_and_basis": "Additional risk",
    "nuclear_contract_counterparty": "LCCC and ESC operating levies",
    "procurement_transaction_cost": "Transaction costs",
    "procurement_day_ahead_rehedging": "Day-ahead re-hedging",
    "procurement_forward_premium": "Forward premium",
    "renewables_obligation": "Renewables Obligation",
    "feed_in_tariffs": "Feed-in Tariffs",
    "sizewell_c_rab": "Sizewell C RAB",
    "warm_home_discount": "Warm Home Discount",
    "battery_revenue_top_up": "Battery revenue top-up",
    "bics_levy": "Network Charging Compensation",
    "aahedc": "AAHEDC",
    "eco": "ECO",
    "duos": "DUoS",
    "tnuos": "TNUoS",
    "bsuos_constraints": "BSUoS: constraints",
    "bsuos_other_services": "BSUoS: other services",
    "ldes_cap_and_floor": "LDES cap and floor",
    "bsuos_voltage_and_stability_constraints": "BSUoS: voltage and stability",
    "interconnector_cap_and_floor_pass_through": "Interconnector cap and floor, credit",
    "supplier_costs_and_margin": "Supplier costs and margin",
    "smart_meter_net_cost_change": "Smart-meter net cost change",
}
KIND_ORDER = ("modelled", "model-informed", "published")
KIND_MEANING = {
    "modelled": "The quantity and its price both come from the run",
    "model-informed": "A published rule applied to a modelled quantity",
    "published": "A published figure, projected to calendar 2030",
}
#: The wholesale group of the worked example, in the order Ofgem builds direct fuel.
WHOLESALE_ALLOWANCES = [
    ("procurement_losses", "Losses"),
    ("procurement_imbalance", "Imbalance"),
    ("procurement_additional_risk_and_basis", "Additional risk"),
    ("procurement_transaction_cost", "Transaction costs"),
    ("procurement_day_ahead_rehedging", "Day-ahead re-hedging"),
    ("procurement_forward_premium", "Forward premium"),
]
SENSITIVITY_LABELS = [
    ("shaping", "low_1pct", "Shaping at 1%"),
    ("shaping", "high_7pct", "Shaping at 7%"),
    ("shaping", "model_profile_only", "Model profile only, no shaping allowance"),
    ("forward_premium", "minus_3pct", "Forward premium -3%"),
    ("forward_premium", "plus_5pct", "Forward premium +5%"),
    ("forward_premium", "stress_plus_10pct", "Forward premium +10% (stress)"),
]
CFD_TECHNOLOGY_LABELS = {
    "wind_offshore": "Offshore wind",
    "nuclear": "Hinkley Point C",
    "wind_onshore": "Onshore wind",
    "biomass": "Biomass (Drax bridging; MGT Teesside Investment Contract)",
    "solar_pv": "Solar",
    "CCS Biomass": "Power BECCS (its own line)",
    "marine": "Tidal stream",
}
TECHNOLOGY_WORDS = {"offshore": "offshore", "onshore": "onshore", "solar": "solar", "tidal": "tidal",
                    "nuclear": "nuclear"}
#: Rounds whose contracts carry the AR4 negative-price term, the modelled AR8 round included.
AR4_PLUS_ROUNDS = ("AR4", "AR5", "AR6", "AR7", "AR7a", MODELLED_ROUND)
CM_CATEGORY_LABELS = {
    "existing": "Existing plant",
    "refurbishing": "Refurbishing plant",
    "interconnector": "Interconnectors",
    "ldes_cap_and_floor": "New long-duration storage (cap and floor)",
    "dsr": "Demand-side response",
    "new_build": "New build",
}
CM_CASE_LABELS = {
    "central": "Central",
    "no_fresh_new_build": "No fresh new build",
    "double_new_build": "Double the fresh new build",
    "storage_efc_low": "Storage de-rated at NESO's 2027/28 factors",
    "ldes_excluded": "New LDES cannot enter",
    "dsr_low": "Low demand-side response",
    "dsr_high": "High demand-side response",
}
RO_CARRIERS = [
    (("wind_offshore",), "Offshore wind"),
    (("wind_onshore",), "Onshore wind"),
    (("solar_pv",), "Solar"),
    (("biomass",), "Biomass"),
    (("marine", "large_hydro", "small_hydro"), "Marine and hydro"),
]
SUPPLIER_COMPONENT_LABELS = {
    "cost to serve": "Core operating cost (cost to serve)",
    "debt related costs": "Debt-related costs",
    "industry charges": "Industry charges (DCC, SEGB, Elexon, RECCo, DCUSA)",
    "ebit": "EBIT",
    "headroom": "Headroom",
}


class DocsInputError(RuntimeError):
    """An input is missing, malformed or from a different run than the bill summary."""


# ── Formatting ───────────────────────────────────────────────────────────────
def num(x: float, dp: int = 2) -> str:
    """``1,039.09``; never ``-0.00``."""
    text = f"{float(x):,.{dp}f}"
    return text[1:] if text.startswith("-") and float(text.replace(",", "")) == 0 else text


def signed(x: float, dp: int = 2) -> str:
    """``+4.41`` / ``-5.74`` (zero unsigned)."""
    text = num(x, dp)
    return text if text.startswith("-") or float(text.replace(",", "")) == 0 else f"+{text}"


def gbp(x: float, dp: int = 2) -> str:
    """``£148.26`` / ``-£2.59``."""
    text = num(x, dp)
    return f"-£{text[1:]}" if text.startswith("-") else f"£{text}"


def round_to(x: float, step: float) -> float:
    return step * round(x / step)


def plain(x: float) -> str:
    """A rounded figure without trailing zeros (``4``, ``4.5``)."""
    return f"{x:,.1f}".rstrip("0").rstrip(".")


def nice_date(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


def table(header: list[str], align: str, rows: list[list[str]]) -> str:
    """A pipe table; ``align`` is one character per column (``l`` or ``r``)."""
    rule = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def bold(text: str) -> str:
    return f"**{text}**" if text else text


# ── Inputs ───────────────────────────────────────────────────────────────────
def display_path(path: Path) -> str:
    """A repository-relative path, also for a file read from another checkout's resources/."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        parts = path.as_posix().split("/")
        for anchor in ("resources", "data"):
            if anchor in parts:
                return "/".join(parts[len(parts) - 1 - parts[::-1].index(anchor):])
        return path.name


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Inputs:
    paths: dict[str, Path]
    components: pd.DataFrame
    summary: dict
    hourly: pd.DataFrame
    ro_summary: dict
    ro_fleet: pd.DataFrame
    balancing: dict
    cm_calendar: dict
    cm_auction: dict
    public_account: dict
    recovery: dict
    line_sources: pd.DataFrame
    cm_supply: pd.DataFrame | None
    run_date: date
    mtimes: dict[str, datetime] = field(default_factory=dict)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_inputs(paths: dict[str, Path], run_date: date | None = None) -> Inputs:
    missing = [f"{key}: {path}" for key, path in paths.items()
               if key not in OPTIONAL_INPUTS and not Path(path).is_file()]
    if missing:
        raise DocsInputError("missing bill-docs inputs (run the bill first; no table is written):\n  "
                             + "\n  ".join(missing))
    mtimes = {key: datetime.fromtimestamp(Path(p).stat().st_mtime) for key, p in paths.items() if Path(p).is_file()}
    summary = _read_json(paths["summary"])
    cm_supply_path = paths.get("cm_supply")
    inputs = Inputs(
        paths={k: Path(v) for k, v in paths.items()},
        components=pd.read_csv(paths["components"]),
        summary=summary,
        hourly=pd.read_csv(paths["hourly"], parse_dates=["snapshot"]),
        ro_summary=_read_json(paths["ro_summary"]),
        ro_fleet=pd.read_csv(paths["ro_fleet"]),
        balancing=_read_json(paths["balancing"]),
        cm_calendar=_read_json(paths["cm_calendar"]),
        cm_auction=_read_json(paths["cm_auction"]),
        public_account=_read_json(paths["public_account"]),
        recovery=_read_json(paths["recovery"]),
        line_sources=pd.read_csv(paths["line_sources"]),
        cm_supply=pd.read_csv(cm_supply_path) if cm_supply_path and Path(cm_supply_path).is_file() else None,
        run_date=run_date or mtimes["summary"].date(),
        mtimes=mtimes,
    )
    check_consistency(inputs)
    return inputs


def _close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def check_consistency(inp: Inputs) -> None:
    """Refuse inputs from different runs: every check names the file to rebuild."""
    s, problems = inp.summary, []
    line = inp.components.set_index("component_id")[VALUE]
    if not _close(line.sum(), s["ex_vat_gbp"], 0.005):
        problems.append(f"components sum to {line.sum():.4f}, the summary's ex-VAT bill is {s['ex_vat_gbp']:.4f}")
    recorded = {
        "recovery": (s.get("recovery_denominators") or {}).get("sha256"),
        "balancing": (s.get("balancing_costs") or {}).get("sha256"),
        "ro_summary": (s.get("ro_scheme_cost") or {}).get("sha256"),
        "public_account": (s.get("ldes_cap_and_floor") or {}).get("sha256"),
    }
    for key, digest in recorded.items():
        if digest is None:
            problems.append(f"the bill summary records no sha256 for {key}")
        elif sha256(inp.paths[key]) != digest:
            problems.append(f"{display_path(inp.paths[key])} is not the file the bill read (sha256 differs)")
    cm_household = cm_line(inp, inp.cm_calendar["calendar_2030_cost_gbp"])
    if "capacity_market" in line and not _close(cm_household, line["capacity_market"], 0.01):
        problems.append(f"the Capacity Market summary gives GBP {cm_household:.2f} a household, the bill "
                        f"{line['capacity_market']:.2f}: {display_path(inp.paths['cm_calendar'])} is newer than the bill")
    ro = inp.ro_summary["gross_gbp_per_3100kwh_ex_vat"]
    if "renewables_obligation" in line and not _close(ro, s["ro_scheme_cost"]["gross_gbp_per_3100kwh_ex_vat"], 1e-6):
        problems.append("the RO summary and the bill's RO block disagree")
    bys = pd.DataFrame(s["cfd_strike_provenance"]["by_strike"])
    if not _close(bys["payment_gbp"].sum(), s["cfd_payment_gbp"], 1.0):
        problems.append("the CfD contracts do not add up to the CfD payment")
    quarters = quarter_table(inp)
    for q in _shaping(inp)["quarters"]:
        hourly = quarters.loc[q["quarter"], "base"]
        if not _close(hourly, q["base_price_gbp_per_mwh"], 0.01):
            problems.append(f"Q{q['quarter']} baseload is {hourly:.3f} in the hourly file, "
                            f"{q['base_price_gbp_per_mwh']:.3f} in the summary")
    twice = sorted(set(inp.line_sources.loc[inp.line_sources["component_id"].duplicated(), "component_id"]))
    if twice:
        problems.append(f"bill lines with two kinds in {display_path(inp.paths['line_sources'])}: {twice}")
    missing_kind = sorted(set(inp.components["component_id"]) - set(inp.line_sources["component_id"]))
    if missing_kind:
        problems.append(f"bill lines with no kind in {display_path(inp.paths['line_sources'])}: {missing_kind}")
    if problems:
        raise DocsInputError("the bill-docs inputs are not one run; nothing written:\n  " + "\n  ".join(problems))


# ── Derived pieces ───────────────────────────────────────────────────────────
def lines(inp: Inputs) -> pd.Series:
    return inp.components.set_index("component_id")[VALUE]


def _shaping(inp: Inputs) -> dict:
    return inp.summary["diagnostics"]["wholesale_procurement"]["shaping"]


def gsp_volume(inp: Inputs) -> float:
    """The GSP volume the Capacity Market is spread over (F18: the GSP charging volume)."""
    d = inp.summary["recovery_denominators"]
    return float(d.get("gsp_charging_volume_mwh") or d["gsp_chargeable_volume_mwh"])


def cm_factor(inp: Inputs) -> float:
    return float(inp.cm_calendar["household_peak_factor"]["factor"])


def cm_line(inp: Inputs, cost_gbp: float) -> float:
    """A calendar-2030 Capacity Market cost as the household line (cost x factor / GSP volume x 3.1)."""
    return cost_gbp * cm_factor(inp) / gsp_volume(inp) * HOUSEHOLD_MWH


def quarter_table(inp: Inputs) -> pd.DataFrame:
    """Per calendar quarter of the hourly file: baseload, and hours at or below GBP 1 and about GBP 0."""
    h = inp.hourly
    if not _close(h["hours"].sum(), inp.summary["reported_period_hours"], 1e-6):
        raise DocsInputError(f"{display_path(inp.paths['hourly'])} covers {h['hours'].sum()} hours, the bill "
                             f"{inp.summary['reported_period_hours']}")
    price, quarter = h["price_gbp_per_mwh"], h["snapshot"].dt.quarter
    weight = h["hours"]
    return pd.DataFrame({
        "base": (price * weight).groupby(quarter).sum() / weight.groupby(quarter).sum(),
        "share_at_or_below_1": ((price <= 1.0) * weight).groupby(quarter).sum() / weight.groupby(quarter).sum(),
    })


def peak_block_hours(inp: Inputs) -> float:
    """Hours in the Elexon peak block, weekdays 07:00-19:00 UK local time (snapshots are UTC)."""
    local = inp.hourly["snapshot"].dt.tz_localize("UTC").dt.tz_convert("Europe/London")
    peak = (local.dt.weekday < 5) & (local.dt.hour >= 7) & (local.dt.hour < 19)
    return float(inp.hourly.loc[peak.to_numpy(), "hours"].sum())


#: The weight the bill applied (``ofgem_cap_rule_price``, 6 Oct 2026), and how its basis reads.
APPLIED_WEIGHT = "quarter_weight"
WEIGHT_BASES = {"household_profile_volume": "household-volume weights",
                "declared_quarter_weights": "Ofgem PC1 seasonal weights"}


def weight_key(quarter: dict) -> str:
    if APPLIED_WEIGHT in quarter:
        return APPLIED_WEIGHT
    keys = [k for k in quarter if k.endswith("weight")]
    if len(keys) != 1:
        raise DocsInputError(f"cannot tell the quarter weight in {sorted(quarter)}")
    return keys[0]


def weight_basis(key: str, quarter: dict | None = None) -> str:
    if key == APPLIED_WEIGHT and quarter is not None:
        basis = quarter.get("quarter_weight_basis")
        if basis not in WEIGHT_BASES:
            raise DocsInputError(f"unknown quarter_weight_basis {basis!r}")
        return WEIGHT_BASES[basis]
    return {"domestic_volume_weight": "household-volume weights"}.get(
        key, "PC1 weights" if "pc1" in key.lower() else key.replace("_", " ").replace(" weight", "") + " weights")


def strike_frame(inp: Inputs) -> pd.DataFrame:
    frame = pd.DataFrame(inp.summary["cfd_strike_provenance"]["by_strike"])
    frame["round"] = frame["strike_key"].map(round_of_key)
    return frame


def round_of_key(key: str) -> str:
    """``AR8_modelled_offshore`` -> ``AR8_modelled``; ``AR7_offshore_scotland`` -> ``AR7``."""
    key = str(key)
    if key.startswith(MODELLED_ROUND + "_"):
        return MODELLED_ROUND
    return key.split("_")[0]


def contract_label(key: str, round_: str) -> str:
    technology = key.split("_")[-1]
    if key == "HPC_nuclear":
        return "Hinkley Point C"
    if round_ == MODELLED_ROUND:
        return f"AR8 {technology} (modelled: contracts awarded in Allocation Round 8)"
    if round_ == "Investment Contract":
        return f"Investment Contracts ({technology})"
    return f"{round_} {technology}"


def supplier_parts(inp: Inputs) -> tuple[list[tuple[str, float]], float, str]:
    """Ofgem's supplier allowances as the bill line's method records them: (parts, total, money).

    Since F18 (6 Oct 2026) the method lists the cap-period allowances other than EBIT and
    headroom, ``...(standing plus unit): cost to serve 110.43; ...; industry charges 21.51
    (Oct-Dec 2026 money)``; EBIT and headroom are the summary's ``supplier_margin``. The
    pre-F18 wording (all parts, ``= GBPtotal (money)``) still reads.
    """
    method = str(inp.components.set_index("component_id").loc["supplier_costs_and_margin", "method"])
    match = (re.search(r"kWh[^:]*:\s*(?P<parts>[^=]+?)\s*=\s*GBP(?P<total>\d[\d,]*(?:\.\d+)?)\s*\((?P<money>[^)]*money)\)",
                       method)
             or re.search(r"kWh[^:]*:\s*(?P<parts>[^()]+?)\s*\((?P<money>[^)]*money)\)", method))
    if not match:
        raise DocsInputError("cannot read Ofgem's supplier allowances from the supplier_costs_and_margin method; "
                             "update supplier_parts() to its wording")
    parts = []
    for piece in match["parts"].split(";"):
        m = re.fullmatch(r"\s*(?P<name>[A-Za-z ,-]+?)\s+(?P<value>-?[\d.]+)\s*", piece)
        if not m:
            raise DocsInputError(f"cannot read the supplier allowance {piece!r}")
        parts.append((m["name"].strip().lower(), float(m["value"])))
    total = sum(v for _, v in parts)
    if "total" in match.groupdict() and match.groupdict()["total"] is not None:
        if not _close(total, float(match["total"].replace(",", "")), 0.02):
            raise DocsInputError("the supplier allowances do not add up to their stated total")
    return parts, total, match["money"]


# ── The generator ────────────────────────────────────────────────────────────
class DocsBuilder:
    def __init__(self, inp: Inputs, groupings: Path, grouping_views: Path):
        self.inp = inp
        self.groupings, self.grouping_views = groupings, grouping_views
        self.values: dict[str, str] = {}
        self.tables: dict[str, str] = {}
        self.line = lines(inp)
        self.rebase = float(inp.summary["model_rebasing_factor"])

    # helpers
    def label(self, cid: str) -> str:
        if cid in LINE_LABELS:
            return LINE_LABELS[cid]
        return str(self.inp.components.set_index("component_id").loc[cid, "component"])

    def put(self, name: str, text: str) -> None:
        if name in self.values and self.values[name] != text:
            raise ValueError(f"value {name} set twice")
        self.values[name] = text

    def provenance(self, keys: list[str]) -> str:
        files = [display_path(self.inp.paths[k]) for k in keys]
        stamped = [f"{f} ({self.inp.mtimes[k]:%Y-%m-%d %H:%M})" for f, k in zip(files, keys)]
        names = ", ".join(f"`{Path(f).name}`" for f in files)
        return (f"% GENERATED by scripts/analysis/build_bill_docs_values.py (rule build_bill_docs_values); do not edit.\n"
                f"% Bill run of {self.inp.run_date.isoformat()}. Inputs: {'; '.join(stamped)}\n\n"
                f"*From {names}.*\n\n")

    def emit(self, name: str, keys: list[str], body: str) -> None:
        if name not in TABLES:
            raise ValueError(f"table {name} is not declared in TABLES")
        self.tables[name] = self.provenance(keys) + body

    def build(self) -> None:
        self.general()
        self.index_tables()
        self.wholesale()
        self.cfd()
        self.capacity_market()
        self.renewables_obligation()
        self.ldes_and_battery()
        self.networks()
        self.recovery()
        self.supplier()
        undeclared = set(TABLES) - set(self.tables)
        if undeclared:
            raise ValueError(f"tables declared but not built: {sorted(undeclared)}")

    # general values and every line
    def general(self) -> None:
        s = self.inp.summary
        self.put("run_date", nice_date(self.inp.run_date))
        self.put("bill_ex_vat", num(s["ex_vat_gbp"]))
        self.put("bill_vat", num(s["vat_gbp"]))
        self.put("bill_inc_vat", num(s["total_gbp_inc_vat"]))
        self.put("vat_rate_pct", plain(100 * s["vat_rate"]))
        self.put("rebasing_factor", f"{self.rebase:.4f}")
        for cid, value in self.line.items():
            self.put(f"line_{cid}", num(value))
            self.put(f"line_{cid}_gbp", gbp(value))

    def index_tables(self) -> None:
        frame = bill_groupings.annotate(self.inp.components, path=self.groupings, views_path=self.grouping_views)
        kinds = self.inp.line_sources.set_index("component_id")["source_type"]
        frame["kind"] = frame["component_id"].map(kinds)
        order = {g: i for i, g in enumerate(bill_groupings.group_order("ofgem", path=self.groupings,
                                                                        views_path=self.grouping_views))}
        frame = frame.assign(_g=frame["ofgem_group"].map(order)).sort_values(["_g", VALUE], ascending=[True, False],
                                                                              kind="stable")
        rows = [[r.ofgem_group, f"{self.label(r.component_id)} (`{r.component_id}`)", r.kind, num(r[VALUE])]
                for _, r in frame.iterrows()]
        s = self.inp.summary
        rows += [["", bold("Total, excluding VAT"), "", bold(num(s["ex_vat_gbp"]))],
                 ["", f"VAT at {plain(100 * s['vat_rate'])}%", "", num(s["vat_gbp"])],
                 ["", bold("Total, including VAT"), "", bold(num(s["total_gbp_inc_vat"]))]]
        self.emit("bill_lines", ["components", "summary", "groupings", "line_sources"],
                  table(["Ofgem group", "Line", "Kind", "£"], "lllr", rows))

        totals = {}
        for view in bill_groupings.VIEWS:
            present = set(frame[f"{view}_group"])
            sums = bill_groupings.totals_by_group(self.inp.components, view, path=self.groupings,
                                                  views_path=self.grouping_views)
            totals[view] = [(g, v) for g, v in sums.items() if g in present]
            for g, v in sums.items():
                if g in present:
                    self.put(f"group_{view}_{slug(g)}", num(v))
        rows = []
        for i in range(max(len(t) for t in totals.values())):
            left = totals["ofgem"][i] if i < len(totals["ofgem"]) else ("", None)
            right = totals["functional"][i] if i < len(totals["functional"]) else ("", None)
            rows.append([left[0], "" if left[1] is None else num(left[1]), "",
                         right[0], "" if right[1] is None else num(right[1])])
        self.emit("group_totals", ["components", "groupings"],
                  table(["Ofgem's categories", "£", "", "What each cost pays for", "£"], "lrllr", rows))

        by_kind = frame.groupby("kind")[VALUE].sum()
        shares = {k: round(100 * by_kind.get(k, 0.0) / s["ex_vat_gbp"]) for k in KIND_ORDER}
        rows = []
        for kind in KIND_ORDER:
            members = frame.loc[frame["kind"].eq(kind)].sort_values(VALUE, ascending=False)
            rows.append([kind.capitalize(), KIND_MEANING[kind], "; ".join(self.label(c) for c in members["component_id"]),
                         num(by_kind.get(kind, 0.0)), f"{shares[kind]}%"])
            self.put(f"share_{kind.replace('-', '_')}_pct", str(shares[kind]))
        self.emit("source_kinds", ["components", "line_sources"],
                  table(["Kind", "Meaning", "Lines", "£", "Share of the 2030 bill"], "lllrr", rows))
        retail = [v for g, v in totals["functional"] if g == "Retail"]
        if retail:
            self.put("retail_total", num(retail[0]))

    def wholesale(self) -> None:
        s, shaping = self.inp.summary, _shaping(self.inp)
        scarcity = s["diagnostics"]["wholesale_procurement"].get("scarcity_markup") or s.get("scarcity_markup") or {}
        if scarcity.get("available"):
            self.put("scarcity_uplift", num(scarcity["load_weighted_uplift_gbp_per_mwh"]))
            self.put("scarcity_hours", num(scarcity["hours_scarce"], 0))
        peak = peak_block_hours(self.inp)
        self.put("peak_block_hours", num(peak, 0))
        self.put("peak_block_share_pct", num(100 * peak / s["reported_period_hours"], 0))
        blend = float(shaping["blend_gbp_per_mwh"])
        self.put("blend", num(blend))
        self.put("blend_3dp", num(blend, 3))
        self.put("blend_2025", num(blend * self.rebase))
        self.put("blend_2025_rounded", num(blend * self.rebase, 0))
        self.put("blend_2025_1dp", num(blend * self.rebase, 1))
        self.put("blend_rounded", num(blend, 0))
        self.put("system_price", num(s["wholesale_price_gbp_per_mwh"]))
        self.put("shaping_rate_pct", f"{100 * float(shaping['rate']):.3f}")

        hourly = quarter_table(self.inp)
        key = weight_key(shaping["quarters"][0])
        self.put("quarter_weight_basis", weight_basis(key, shaping["quarters"][0]))
        body = [f"```{{list-table}} Quarterly prices, 2030 (GBP/MWh, 2024 money)",
                ":header-rows: 1", "",
                "* - Quarter", "  - Baseload", "  - Peak", "  - 70/30 blend",
                f"  - Weight ({weight_basis(key, shaping['quarters'][0])})", "  - Hours at or below £1"]
        for q in shaping["quarters"]:
            n = q["quarter"]
            qblend = q.get("blend_gbp_per_mwh", 0.7 * q["base_price_gbp_per_mwh"] + 0.3 * q["peak_price_gbp_per_mwh"])
            body += [f"* - Q{n}", f"  - {num(q['base_price_gbp_per_mwh'])}", f"  - {num(q['peak_price_gbp_per_mwh'])}",
                     f"  - {num(qblend)}", f"  - {100 * q[key]:.1f}%",
                     f"  - {100 * hourly.loc[n, 'share_at_or_below_1']:.0f}%"]
        body.append("```")
        self.emit("wholesale_quarters", ["summary", "hourly"], "\n".join(body) + "\n")

        energy, shaping_line = self.line["direct_energy"], self.line["shaping_adjustment"]
        energy_cost = energy + shaping_line
        rows = [["Energy", f"{num(blend, 3)} × {self.rebase:.4f} × {plain(HOUSEHOLD_MWH)}", num(energy)],
                ["Shaping", f"{100 * shaping_line / energy:.3f}% of energy", num(shaping_line)],
                ["*Energy cost*", "", f"*{num(energy_cost)}*"]]
        total = energy_cost
        for cid, name in WHOLESALE_ALLOWANCES:
            value = self.line.get(cid, 0.0)
            total += value
            calc = (f"{value / energy_cost:.4f} × energy cost" if cid == "procurement_losses"
                    else f"{100 * value / energy_cost:.3f}%")
            if cid == "procurement_forward_premium" and value == 0:
                calc = "0%"
            rows.append([name, calc, num(value)])
        per_mwh = total / HOUSEHOLD_MWH
        rows.append([bold("Total"), f"about £{num(per_mwh, 0)}/MWh all-in", bold(num(total))])
        self.put("wholesale_procurement_total", num(total))
        self.put("wholesale_all_in_per_mwh", num(per_mwh, 0))
        self.put("losses_factor_less_one", f"{self.line['procurement_losses'] / energy_cost:.4f}")
        self.emit("wholesale_build_up", ["components", "summary"], table(["Line", "Calculation", "£"], "llr", rows))

        sens = s["diagnostics"]["wholesale_procurement"]["sensitivities_gbp_per_3100kwh_ex_vat"]
        rows = [[label, signed(sens[group][k])] for group, k, label in SENSITIVITY_LABELS if k in sens.get(group, {})]
        self.emit("wholesale_sensitivities", ["summary"], table(["Sensitivity", "£"], "lr", rows))

    def cfd(self) -> None:
        s, frame = self.inp.summary, strike_frame(self.inp)
        h = self.inp.hourly["price_gbp_per_mwh"]
        self.put("min_price", num(s["minimum_price_gbp_per_mwh"], 4))
        self.put("min_price_gbp", gbp(s["minimum_price_gbp_per_mwh"], 4))
        self.put("hours_below_zero", num(s["hours_price_below_minus_0_01_gbp_per_mwh"], 0))
        self.put("hours_about_zero", num(int((h.abs() <= 0.01).sum()), 0))
        self.put("hours_at_or_below_1", num(s["hours_price_at_or_below_1_gbp_per_mwh"], 0))

        payment, twh = frame["payment_gbp"].sum(), frame["dispatch_mwh"].sum() / 1e6
        topup = payment / (twh * 1e6)
        strike = (frame["strike_price"] * frame["dispatch_mwh"]).sum() / (twh * 1e6)
        # F18: the CfD's own recovery volume (cfd_recovery_denominator_mwh is the volume over the
        # household loss factor on the two-volume basis).
        eligible = float(((s.get("line_recovery") or {}).get("cfd") or {}).get("volume_mwh")
                         or s["cfd_recovery_denominator_mwh"])
        gross = s["diagnostics"]["cfd_line_gbp_per_3100kwh"]
        self.put("cfd_payment_bn", num(payment / 1e9))
        self.put("cfd_payment_bn_1dp", num(payment / 1e9, 1))
        self.put("cfd_twh", num(twh, 1))
        self.put("cfd_twh_int", num(twh, 0))
        self.put("cfd_topup_per_mwh", num(topup, 1))
        self.put("cfd_topup_per_mwh_2dp", num(topup))
        self.put("cfd_avg_strike", num(strike, 0))
        self.put("cfd_capture_price", num(strike - topup, 0))
        self.put("cfd_eligible_twh", num(eligible / 1e6, 1))
        self.put("cfd_eligible_twh_2dp", num(eligible / 1e6))
        self.put("cfd_gross_household", num(gross))
        self.put("cfd_levy_rate", num(gross / HOUSEHOLD_MWH))

        ar8 = frame.loc[frame["round"].eq(MODELLED_ROUND)]
        for technology in ("offshore", "onshore", "solar", "tidal"):
            rows = ar8.loc[ar8["strike_key"].eq(f"{MODELLED_ROUND}_{technology}")]
            self.put(f"cfd_ar8_{technology}_twh", num(rows["dispatch_mwh"].sum() / 1e6, 1))
            strike = rows["strike_price"]
            if len(strike):
                self.put(f"cfd_ar8_{technology}_strike", num(strike.iloc[0]))
        self.put("cfd_ar8_bn", num(ar8["payment_gbp"].sum() / 1e9, 1))

        ar4 = frame.loc[frame["round"].isin(AR4_PLUS_ROUNDS)]
        ar4_twh = round_to(ar4["dispatch_mwh"].sum() / 1e6, 10)
        ar4_strike = round_to((ar4["strike_price"] * ar4["dispatch_mwh"]).sum() / ar4["dispatch_mwh"].sum(), 5)
        at_stake = round_to(ar4_twh / 3 * ar4_strike / 1e3, 0.5)
        self.put("ar4_plus_twh", num(ar4_twh, 0))
        self.put("ar4_plus_strike", num(ar4_strike, 0))
        self.put("ar4_zero_price_bn", plain(at_stake))
        self.put("ar4_zero_price_household",
                 num(round_to(at_stake * 1e9 * self.rebase / eligible * HOUSEHOLD_MWH, 5), 0))

        by_tech = frame.groupby("carrier").agg(mwh=("dispatch_mwh", "sum"), gbp=("payment_gbp", "sum"))
        by_tech = by_tech.sort_values("gbp", ascending=False)
        rows = []
        for carrier, r in by_tech.iterrows():
            t = r.mwh / 1e6
            rows.append([CFD_TECHNOLOGY_LABELS.get(carrier, carrier), num(t, 1 if t >= 1 else 2), num(r.gbp / 1e9),
                         num(r.gbp / r.mwh, 1)])
        rows.append([bold("Total"), bold(num(twh, 1)), bold(num(payment / 1e9)), bold(num(topup, 1))])
        self.emit("cfd_by_technology", ["summary"],
                  table(["Technology", "TWh", "£bn", "Average top-up, £/MWh"], "lrrr", rows))

        threshold = 0.5e9
        big = frame.loc[frame["payment_gbp"] > threshold].sort_values("payment_gbp", ascending=False)
        rows = [[contract_label(r.strike_key, r["round"]), num(r.strike_price), num(r.dispatch_mwh / 1e6, 1),
                 num(r.payment_gbp / 1e9)] for _, r in big.iterrows()]
        self.put("cfd_largest_threshold_bn", plain(threshold / 1e9))
        self.emit("cfd_largest_contracts", ["summary"],
                  table(["Contract", "Strike, 2024 money", "TWh", "£bn"], "lrrr", rows))

        oldest = frame.loc[frame["strike_key"].isin(["Investment Contract_offshore", "AR1_offshore"])]
        self.put("cfd_oldest_offshore_output_pct", num(100 * oldest["dispatch_mwh"].sum() / (twh * 1e6), 0))
        self.put("cfd_oldest_offshore_payment_pct", num(100 * oldest["payment_gbp"].sum() / payment, 0))

    def capacity_market(self) -> None:
        cal, auction = self.inp.cm_calendar, self.inp.cm_auction
        years = cal["delivery_years"]
        dy_a, dy_b = sorted(years)
        a, b = years[dy_a], years[dy_b]
        held = auction.get("checks", {}).get(f"T-4 {short_dy(dy_a)}", {})

        def bn(x):
            return f"£{num(x / 1e9)}bn"

        t4_a = (bn(a["t4_modelled_gbp"]) + ", modelled" if "t4_modelled_gbp" in a
                else f"(in the contract book: held March 2026, £{num(held.get('actual_price', float('nan')))}/kW)")
        t4_b = bn(b["t4_modelled_gbp"]) + ", modelled" if "t4_modelled_gbp" in b else "(in the contract book)"
        rows = [["Contract book: agreements already signed", bn(a["contract_book_gbp"]), bn(b["contract_book_gbp"])],
                ["T-4 auction", t4_a, t4_b],
                ["T-1 auction (1.0 GW at the year's T-4 price)", bn(a.get("t1_gbp", 0.0)), bn(b.get("t1_gbp", 0.0))],
                [bold("Delivery year"), bold(bn(a["total_gbp"])), bold(bn(b["total_gbp"]))],
                ["Weight in calendar 2030", plain(a["weight_in_calendar_2030"]), plain(b["weight_in_calendar_2030"])],
                [bold("Calendar 2030"), "", bold(bn(cal["calendar_2030_cost_gbp"]))]]
        self.emit("cm_delivery_years", ["cm_calendar", "cm_auction"],
                  table(["", short_dy(dy_a), short_dy(dy_b)], "lrr", rows) + f"\nAll in {cal['price_year']} money.\n")
        for key, value in (("cm_book_a_bn", a["contract_book_gbp"]), ("cm_book_b_bn", b["contract_book_gbp"]),
                           ("cm_t1_a_bn", a.get("t1_gbp", 0.0)), ("cm_t1_b_bn", b.get("t1_gbp", 0.0)),
                           ("cm_t4_b_bn", b.get("t4_modelled_gbp", 0.0)), ("cm_calendar_bn", cal["calendar_2030_cost_gbp"])):
            self.put(key, num(value / 1e9))

        factor = cal["household_peak_factor"]
        self.put("cm_household_factor", f"{factor['factor']:.4f}")
        self.put("cm_ofgem_allowance_2024", num(factor["ofgem_allowance_gbp_per_3100kwh_nominal"]))
        self.put("cm_flat_share_2024", num(factor["flat_share_gbp_per_3100kwh"]))
        self.put("cm_gsp_twh", num(gsp_volume(self.inp) / 1e6, 1))
        for name, check in auction.get("checks", {}).items():
            tag = name.replace("T-4 ", "").replace("/", "_")
            self.put(f"cm_check_{tag}_modelled", num(check["modelled_price"]))
            self.put(f"cm_check_{tag}_actual", num(check["actual_price"]))

        cases = auction["t4_2030_31"]
        central = cases["central"]
        self.put("cm_t4_central_price", num(central["price"]))
        self.put("cm_t4_supply_gw", num(central["entered_total_mw"] / 1e3, 1))
        entered = pd.Series(central["entered_mw"]).sort_values(ascending=False)
        if self.inp.cm_supply is not None and "case" in self.inp.cm_supply:
            supply = self.inp.cm_supply.loc[self.inp.cm_supply["case"].eq("central")]
            if not _close(supply["supply_mw"].sum(), central["entered_total_mw"], 1.0):
                raise DocsInputError("cm_t4_2030_31_supply.csv and cm_auction_model.json disagree on central supply")
        rows = [[CM_CATEGORY_LABELS.get(k, k), num(v, 0)] for k, v in entered.items()]
        rows.append([bold("Total"), bold(num(central["entered_total_mw"], 0))])
        self.emit("cm_supply", ["cm_auction"], table(["Supply, MW de-rated", "Central"], "lr", rows))

        costs = cal["calendar_2030_cost_by_t4_case_gbp"]
        rows, prices, totals = [], [], []
        for case, result in cases.items():
            if case not in costs:
                raise DocsInputError(f"auction case {case} has no calendar-2030 cost in {display_path(self.inp.paths['cm_calendar'])}")
            price = num(result["price"]) + (" (cap, short)" if result.get("shortfall") else "")
            row = [CM_CASE_LABELS.get(case, case), price, bn(costs[case]), num(cm_line(self.inp, costs[case]))]
            rows.append([bold(c) for c in row] if case == "central" else row)
            prices.append(result["price"])
            totals.append(costs[case])
        self.put("cm_case_price_min", num(min(prices), 0))
        self.put("cm_case_price_max", num(max(prices), 0))
        self.put("cm_case_cost_min_bn", num(min(totals) / 1e9))
        self.put("cm_case_cost_max_bn", num(max(totals) / 1e9))
        self.emit("cm_cases", ["cm_auction", "cm_calendar", "summary"],
                  table(["Case", "Clearing price, £/kW-year", "Calendar 2030", "£ a household"], "lrrr", rows))

    def renewables_obligation(self) -> None:
        ro, fleet = self.inp.ro_summary, self.inp.ro_fleet
        value, share = ro["notional_roc_value"], ro["uk_obligation_split"]["gb_share_of_uk_obligation"]
        roc_value = value["notional_roc_value_gbp_per_roc_report_prices"]
        put = self.put
        put("ro_register_gw", num(ro["register_total_capacity_mw"] / 1e3, 1))
        put("ro_expired_gw", num(ro["expired_capacity_mw"] / 1e3, 1))
        put("ro_part_year_gw", num(ro.get("unaccredited_part_of_year_capacity_mw", 0.0) / 1e3, 1))
        bics = ((self.inp.summary.get("levy_exemptions") or {}).get("sensitivities") or {}).get("bics_2027")
        if bics:
            exempt = float(bics["additional_exempt_share"])
            put("ro_bics_sensitivity", num(ro["gross_gbp_per_3100kwh_ex_vat"] * exempt / (1 - exempt)))
            put("bics_exempt_share_pct", plain(100 * exempt))
        put("ro_inactive_gw", num(ro["inactive_capacity_mw"] / 1e3, 1))
        put("ro_eligible_gw", num(ro["eligible_capacity_mw"] / 1e3, 1))
        put("ro_stations", num(ro["eligible_stations"], 0))
        put("ro_ni_gw", num(ro["roc_volume_by_jurisdiction"]["NI"]["capacity_mw"] / 1e3, 1))
        put("ro_value", num(roc_value))
        put("ro_base_year", str(value["base_obligation_year"]))
        put("ro_base_buyout", num(value["base_buyout_price_gbp_per_roc"]))
        put("ro_base_recycle", num(value["base_recycle_value_gbp_per_roc"]))
        put("ro_gb_share_pct", num(100 * share))
        put("ro_supply_twh", num(ro["obligation_eligible_supply_mwh"] / 1e6, 1))
        put("ro_cost_bn", num(ro["gross_scheme_cost_gbp"] / 1e9))
        put("ro_uk_cost_bn", num(ro["uk_roc_volume"] * roc_value / 1e9))
        put("ro_household", num(ro["gross_gbp_per_3100kwh_ex_vat"]))
        relief = ro["relief_cases"]
        put("ro_relief_line", num(relief["relief_extended"]["gbp_per_3100kwh_ex_vat"]))
        put("ro_relief_swing", num(relief["relief_ends_march_2029"]["gbp_per_3100kwh_ex_vat"]
                                   - relief["relief_extended"]["gbp_per_3100kwh_ex_vat"]))
        f = fleet.set_index("carrier")
        for carrier, tag in (("wind_offshore", "offshore"), ("wind_onshore", "onshore"), ("solar_pv", "solar"),
                             ("biomass", "biomass")):
            if carrier in f.index:
                put(f"ro_lf_{tag}", num(f.loc[carrier, "applied_load_factor"], 3 if tag != "biomass" else 2))
        if "biomass" in f.index:
            put("ro_lf_biomass_model", num(f.loc["biomass", "pypsa_load_factor"], 3))

        rows, total = [], {"mw": 0.0, "mwh": 0.0, "rocs": 0.0}
        for carriers, label in RO_CARRIERS:
            part = fleet.loc[fleet["carrier"].isin(carriers)]
            if part.empty:
                continue
            mw, mwh, rocs = part["eligible_capacity_mw"].sum(), part["estimated_generation_mwh"].sum(), part["roc_volume"].sum()
            single = len(part) == 1
            rows.append([label, num(mw, 0),
                         num(part["applied_load_factor"].iloc[0], 3) if single else "",
                         num(mwh / 1e6, 1 if mwh >= 1e6 else 2),
                         num(rocs / mwh) if single else "",
                         num(rocs / 1e6, 1), num(rocs * roc_value * share / 1e9)])
            total["mw"] += mw; total["mwh"] += mwh; total["rocs"] += rocs
        unlisted = sorted(set(fleet["carrier"]) - {c for cs, _ in RO_CARRIERS for c in cs})
        if unlisted:
            raise DocsInputError(f"RO carriers with no row in the docs table: {unlisted}")
        cost = total["rocs"] * roc_value * share
        if not _close(cost, ro["gross_scheme_cost_gbp"], 1e3):
            raise DocsInputError("the RO fleet does not add up to the RO summary's scheme cost")
        rows.append([bold("Total"), bold(num(total["mw"], 0)), "", bold(num(total["mwh"] / 1e6, 1)), "",
                     bold(num(total["rocs"] / 1e6, 1)), bold(num(cost / 1e9))])
        self.emit("ro_by_carrier", ["ro_fleet", "ro_summary"],
                  table(["Carrier", "Eligible MW", "Load factor", "TWh", "Banding", "Million ROCs", "£bn"],
                        "lrrrrrr", rows))

    def ldes_and_battery(self) -> None:
        private = self.inp.public_account["cases"]["private"]
        self.put("ldes_topup_bn", num(private["ldes_floor_topup_gbp"] / 1e9))
        self.put("ldes_topup_at_wacc_bn", num(private["ldes_floor_topup_at_wacc_gbp"] / 1e9))
        self.put("ldes_floor_rate_pct", plain(100 * private["ldes_floor_real_rate"]))
        self.put("ldes_cm_netted_m", num(private["ldes_capacity_market_revenue_netted_gbp"] / 1e6, 1))
        self.put("battery_top_up_m", num(private["battery_revenue_top_up_gbp"] / 1e6, 0))

    def networks(self) -> None:
        b, s = self.inp.balancing["annual"], self.inp.summary
        metered = float(s["recovery_denominators"]["metered_supply_mwh"])
        page_lines = ["duos", "tnuos", "bsuos_constraints", "bsuos_other_services",
                      "bsuos_voltage_and_stability_constraints", "interconnector_cap_and_floor_pass_through"]
        self.put("network_lines_total", num(sum(self.line.get(c, 0.0) for c in page_lines)))
        slack = b.get("feasibility_slack_cost_in_accepted_costs_gbp", b.get("feasibility_slack_cost_gbp", 0.0))
        rows = [["Accepted offers (plant turned up)", num(b["accepted_offer_cost_gbp"] / 1e6, 1)],
                ["Accepted bids (plant turned down)", num(b["accepted_bid_cost_gbp"] / 1e6, 1)],
                ["less feasibility slack (unserved load at VOLL)", num(-slack / 1e6, 1)],
                ["less turn-down of non-participants at the LP's penalty bid",
                 num(-b.get("non_participant_turn_down_cost_at_penalty_gbp", 0.0) / 1e6, 1)]]
        if b.get("tie_break_step_cost_gbp"):
            rows.append(["less tie-break steps", num(-b["tie_break_step_cost_gbp"] / 1e6, 1)])
        if b.get("include_interconnector_retrading"):
            rows.append(["plus interconnector re-trading", num(b["interconnector_retrading_cost_gbp"] / 1e6, 1)])
        rebuilt = (b["accepted_offer_cost_gbp"] + b["accepted_bid_cost_gbp"] - slack
                   - b.get("non_participant_turn_down_cost_at_penalty_gbp", 0.0) - b.get("tie_break_step_cost_gbp", 0.0)
                   + (b["interconnector_retrading_cost_gbp"] if b.get("include_interconnector_retrading") else 0.0))
        if not _close(rebuilt, b["constraint_cost_gbp"], 1e3):
            raise DocsInputError("the constraint build-up does not reproduce the balancing summary's constraint cost")
        rows.append([bold("Constraint cost"), bold(num(b["constraint_cost_gbp"] / 1e6, 1))])
        self.emit("network_constraint_build_up", ["balancing"],
                  table([f"Part, 2030 run", f"£m, {b.get('model_price_year', 2024)} money"], "lr", rows))
        put = self.put
        put("constraint_m", num(b["constraint_cost_gbp"] / 1e6, 0))
        put("constraint_bn", num(b["constraint_cost_gbp"] / 1e9))
        put("thermal_redispatch_bn", num(b["constraint_cost_excluding_retrading_gbp"] / 1e9))
        put("retrading_bn", num(b["interconnector_retrading_cost_gbp"] / 1e9))
        put("cfd_turndown_m", num(b["cfd_curtailment_compensation_gbp"] / 1e6, 1))
        put("cfd_turndown_twh", num(b["cfd_curtailed_mwh"] / 1e6, 1))
        put("metered_twh", num(metered / 1e6))
        levy = s["diagnostics"].get("levy_and_cm_sensitivities_gbp_per_3100kwh_ex_vat", {})
        if "duos" in self.line and "duos_low" in levy:
            put("duos_low", num(self.line["duos"] + levy["duos_low"]))
            put("duos_high", num(self.line["duos"] + levy["duos_high"]))

    def recovery(self) -> None:
        d = self.inp.summary["recovery_denominators"]
        if d.get("recovery_basis") == "loss_inclusive_two_volume":
            self.recovery_two_volume(d)
            return
        twh = {k: float(d[k]) / 1e6 for k in ("gsp_chargeable_volume_mwh", "gb_electrical_load_mwh",
                                               "electrolyser_draw_mwh", "metered_supply_mwh",
                                               "cfd_eligible_supplier_sales_mwh", "h2_offtake_mwh")}
        loss, share = float(d["gsp_to_meter_loss_factor"]), float(d["cfd_eligible_share"])
        put = self.put
        put("gsp_twh", num(twh["gsp_chargeable_volume_mwh"]))
        put("load_twh", num(twh["gb_electrical_load_mwh"]))
        put("electrolyser_twh", num(twh["electrolyser_draw_mwh"]))
        put("eligible_twh", num(twh["cfd_eligible_supplier_sales_mwh"]))
        put("h2_twh", num(twh["h2_offtake_mwh"]))
        put("loss_factor", f"{loss:.4f}")
        put("eligible_share", f"{share:.4f}")
        put("eii_exemption_pct", f"{100 * (1 - share):.2f}")
        rows = [["GSP chargeable volume", num(twh["gsp_chargeable_volume_mwh"]),
                 f"GB electrical load ({num(twh['gb_electrical_load_mwh'])}) plus the electricity electrolysers draw "
                 f"({num(twh['electrolyser_draw_mwh'])}), at the grid supply point", "Capacity Market"],
                ["Metered supply", num(twh["metered_supply_mwh"]), f"GSP volume ÷ loss factor ({loss:.4f})",
                 "BSUoS (constraints, other services, voltage and stability), LDES cap and floor"],
                ["Eligible supply", num(twh["cfd_eligible_supplier_sales_mwh"]),
                 f"Metered supply × {share:.4f}, the share left after the energy-intensive-industry exemption "
                 "(LCCC 2024-25 reconciled data)",
                 "CfD, Power BECCS CfD, Net Zero Teesside DPA, Feed-in Tariffs, Renewables Obligation"]]
        self.emit("recovery_volumes", ["summary", "recovery"],
                  table(["Volume", "TWh", "How it is formed", "Lines recovered over it"], "lrll", rows))

        s = self.inp.summary
        payment_bn = s["cfd_payment_gbp"] / 1e9
        eligible = twh["cfd_eligible_supplier_sales_mwh"]
        gross = s["diagnostics"]["cfd_line_gbp_per_3100kwh"]
        rate = gross / HOUSEHOLD_MWH
        body = ("$$\n"
                f"\\frac{{{num(payment_bn)}\\ \\text{{£bn}} \\times {self.rebase:.4f}}}{{{num(eligible, 1)}\\ \\text{{TWh}}}}"
                f" = {num(rate)}\\ \\text{{£/MWh}}\n\\qquad\n"
                f"{num(rate)}\\ \\text{{£/MWh}} \\times {plain(HOUSEHOLD_MWH)}\\ \\text{{MWh}} = \\text{{£}}{num(gross)}\n"
                "$$\n")
        self.emit("recovery_cfd_example", ["summary"], body)

    def recovery_two_volume(self, d) -> None:
        """F18: the two recovery volumes, their factors and the CfD worked example."""
        twh = {k: float(d[k]) / 1e6 for k in ("metered_supply_mwh", "gsp_charging_volume_mwh",
                                               "cfd_eligible_gsp_volume_mwh", "ro_eligible_metered_supply_mwh",
                                               "gb_electrical_load_mwh", "electrolyser_draw_mwh", "h2_offtake_mwh")}
        to_metered, to_gsp = float(d["system_to_metered_factor"]), float(d["metered_to_gsp_factor"])
        loss, share = float(d["household_loss_factor"]), float(d["cfd_eligible_share"])
        put = self.put
        put("load_twh", num(twh["gb_electrical_load_mwh"]))
        put("electrolyser_twh", num(twh["electrolyser_draw_mwh"]))
        put("h2_twh", num(twh["h2_offtake_mwh"]))
        put("metered_supply_twh", num(twh["metered_supply_mwh"]))
        put("gsp_twh", num(twh["gsp_charging_volume_mwh"]))
        put("cfd_eligible_gsp_twh", num(twh["cfd_eligible_gsp_volume_mwh"]))
        put("ro_eligible_twh", num(twh["ro_eligible_metered_supply_mwh"]))
        put("eligible_twh", num(twh["cfd_eligible_gsp_volume_mwh"]))
        put("system_to_metered_factor", f"{to_metered:.3f}")
        put("metered_to_gsp_factor", f"{to_gsp:.3f}")
        put("household_loss_factor", f"{loss:.4f}")
        put("loss_factor", f"{loss:.4f}")
        put("eligible_share", f"{share:.4f}")
        put("eii_exemption_pct", f"{100 * (1 - share):.2f}")
        load = num(twh["gb_electrical_load_mwh"])
        draw = num(twh["electrolyser_draw_mwh"])
        rows = [["Metered supply", num(twh["metered_supply_mwh"]),
                 f"GB electrical load ({load}) ÷ {to_metered:.3f}, plus the electricity electrolysers draw ({draw})",
                 "Battery revenue top-up; every per-household line's system equivalent; the Exchequer's per-household key"],
                ["Eligible metered supply", num(twh["ro_eligible_metered_supply_mwh"]),
                 f"Metered supply × {share:.4f}, the share left after the energy-intensive-industry exemption",
                 "Renewables Obligation, Feed-in Tariffs"],
                ["GSP charging volume", num(twh["gsp_charging_volume_mwh"]),
                 f"GB electrical load ÷ {to_metered:.3f} × {to_gsp:.3f}, plus the electrolysers' draw",
                 f"BSUoS (three lines), LDES cap and floor, AAHEDC, LCCC/ESC (each × {loss:.4f}); Capacity Market "
                 "(with its own household factor)"],
                ["Eligible GSP volume", num(twh["cfd_eligible_gsp_volume_mwh"]),
                 f"GSP charging volume × {share:.4f}",
                 f"CfD, Power BECCS CfD, Net Zero Teesside DPA (each × {loss:.4f})"]]
        self.emit("recovery_volumes", ["summary", "recovery"],
                  table(["Volume", "TWh", "How it is formed", "Lines recovered over it"], "lrll", rows))

        s = self.inp.summary
        payment_bn = s["cfd_payment_gbp"] / 1e9
        gross = s["diagnostics"]["cfd_line_gbp_per_3100kwh"]
        rate = gross / HOUSEHOLD_MWH / loss
        body = ("$$\n"
                f"\\frac{{{num(payment_bn)}\\ \\text{{£bn}} \\times {self.rebase:.4f}}}"
                f"{{{num(twh['cfd_eligible_gsp_volume_mwh'], 1)}\\ \\text{{TWh}}}}"
                f" = {num(rate)}\\ \\text{{£/MWh}}\n\\qquad\n"
                f"{num(rate)}\\ \\text{{£/MWh}} \\times {plain(HOUSEHOLD_MWH)}\\ \\text{{MWh}} \\times {loss:.4f}"
                f" = \\text{{£}}{num(gross)}\n"
                "$$\n")
        self.emit("recovery_cfd_example", ["summary"], body)

    def supplier(self) -> None:
        parts, total, money = supplier_parts(self.inp)
        line = self.line["supplier_costs_and_margin"]
        margin = self.inp.summary.get("supplier_margin")
        if margin is None or margin.get("rule") is None:
            # Before F18: every allowance in cap-period cash, rebased as one.
            rows = [[SUPPLIER_COMPONENT_LABELS.get(name, name.capitalize()), num(value)] for name, value in parts]
            rows.append([bold("Total"), bold(num(total))])
            self.emit("supplier_components", ["components"], table(["Ofgem component", f"£, {money}"], "lr", rows))
            self.put("supplier_ofgem_total", num(total))
            self.put("supplier_to_report_money", f"{line / total:.4f}")
        else:
            # F18: the allowances and levelisation rebased; EBIT and headroom on this bill's own costs.
            factor = float(margin["external_factor"])
            rule, base = margin["rule"], margin["base"]
            levelisation = float(margin["levelisation_gbp"])
            ebit, headroom = float(margin["ebit_gbp"]), float(margin["headroom_gbp"])
            rows = [[SUPPLIER_COMPONENT_LABELS.get(name, name.capitalize()), num(value), num(value * factor)]
                    for name, value in parts]
            rows.append(["Direct-debit levelisation (Annex 9)", num(levelisation / factor), num(levelisation)])
            rows.append([f"EBIT: {100 * float(rule['ebit_variable_share']):.4f}% of £{num(base['costs_gbp'])} of "
                         f"costs + £{num(float(margin['ebit_fixed_gbp']))}", "", num(ebit)])
            rows.append([f"Headroom: {100 * float(rule['headroom_share']):.4f}% of (£"
                         f"{num(base['costs_excluding_networks_gbp'])} excluding networks + EBIT)", "", num(headroom)])
            total_report = total * factor + levelisation + ebit + headroom
            if not _close(total_report, line, 0.02):
                raise DocsInputError(f"the supplier parts make £{total_report:.2f}, the line £{line:.2f}")
            rows.append([bold("Total"), "", bold(num(line))])
            self.emit("supplier_components", ["components", "summary"],
                      table(["Part", f"£, {money}", "£, 2025 money"], "lrr", rows))
            self.put("supplier_ofgem_total", num(total))
            self.put("supplier_to_report_money", f"{factor:.4f}")
            self.put("supplier_levelisation", num(levelisation))
            self.put("supplier_ebit", num(ebit))
            self.put("supplier_headroom", num(headroom))
            self.put("supplier_cap_ebit", num(float(margin["cap_period_ebit_gbp"])))
            self.put("supplier_cap_headroom", num(float(margin["cap_period_headroom_gbp"])))
            self.put("supplier_margin_change", gbp(float(margin["change_against_cap_period_cash_gbp"])))
            self.put("supplier_cost_base", num(float(base["costs_gbp"])))
            self.put("supplier_cost_base_ex_networks", num(float(base["costs_excluding_networks_gbp"])))
        smncc = re.search(r"=\s*GBP(-?\d+(?:\.\d+)?)", str(self.inp.components.set_index("component_id")
                                                  .loc["smart_meter_net_cost_change", "method"]))
        if smncc:
            self.put("smncc_cap_money", gbp(float(smncc.group(1))))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def short_dy(delivery_year: str) -> str:
    """``2029/2030`` -> ``2029/30``."""
    a, _, b = delivery_year.partition("/")
    return f"{a}/{b[-2:]}" if b else delivery_year


def write_outputs(builder: DocsBuilder, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in builder.tables.items():
        path = out_dir / f"{name}.md"
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    inp = builder.inp
    payload = {
        "_meta": {
            "generator": "scripts/analysis/build_bill_docs_values.py",
            "rule": "build_bill_docs_values",
            "bill_run_date": inp.run_date.isoformat(),
            "inputs": {k: {"path": display_path(p), "modified": inp.mtimes[k].isoformat(timespec="seconds")}
                       for k, p in inp.paths.items() if k in inp.mtimes},
            "note": "Loaded by docs/source/conf.py as myst_substitutions; use {{ name }} in the bill chapter.",
        },
        "values": dict(sorted(builder.values.items())),
    }
    path = out_dir / VALUES_FILE
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    written.append(path)
    return written


def build(paths: dict[str, Path], out_dir: Path, run_date: date | None = None) -> list[Path]:
    inp = load_inputs(paths, run_date)
    builder = DocsBuilder(inp, paths["groupings"], paths["grouping_views"])
    builder.build()
    return write_outputs(builder, out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for key, default in {**DEFAULT_INPUTS, **OPTIONAL_INPUTS}.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=str(default),
                            help=f"default {display_path(default)}" + (" (read when present)" if key in OPTIONAL_INPUTS else ""))
    parser.add_argument("--out-dir", default=str(OUT_DIR), help=f"default {display_path(OUT_DIR)}")
    parser.add_argument("--run-date", type=date.fromisoformat,
                        help="the bill run's date (default: the bill summary's modification date)")
    args = parser.parse_args(argv)
    paths = {key: Path(getattr(args, key)) for key in {**DEFAULT_INPUTS, **OPTIONAL_INPUTS}}
    try:
        written = build(paths, Path(args.out_dir), args.run_date)
    except DocsInputError as exc:
        raise SystemExit(f"build_bill_docs_values: {exc}") from None
    print(f"wrote {len(written)} files to {display_path(Path(args.out_dir))}")


if __name__ == "__main__":
    main()
