"""Support lines for the 2050 private counterfactual's unsupported plant (FES 2050 decision 58, 3 Oct 2026).

The review found plant that the 2050 private case leaves short of its costs: batteries,
energy from waste, unabated biomass and expired offshore wind (finding 6). The user chose to
support each one. Every line here is a payment to an asset row, so it reaches the row's owner
through its producer rent and the state's share of it (``public_surplus_before_recycling_gbp``),
whoever that is in each case; it sits in the bill ledger and in the three-party identity
through the register (market cases) and the acquisition tariff (P3).

* **Ancillary services** (``ancillary_services_revenue_gbp``): NESO's frequency response,
  reserve, stability, reactive power and restoration, sized from NESO's FY2024/25 outturn and
  projected to 2050 (``data/public_ownership/financing/ancillary_services_2050.csv``, sources in
  the file), allocated only to the technologies that provide each service by their participation
  share, then to each provider's register rows pro rata to capacity. The bill pays these
  services: ``bsuos_other_services`` (NESO's non-constraint BSUoS) and
  ``bsuos_voltage_and_stability_constraints`` (the voltage and inertia actions). Decision 65
  (3 Oct 2026) rebases both 2050 lines on this table (``build_2050_bill_inputs.py``): the
  services part of each line is :func:`services_held_by_line` (the same table and case), so the
  services paid to providers, in the register and outside it, equal the services the lines hold
  exactly; NESO's internal cost and the other trade categories stay demand-scaled. So the revenue
  is a transfer inside those lines. Providers outside the register (demand side, interconnectors,
  synchronous condensers, other) are paid their share from the line outside the register. The
  model co-optimises no reserve, response, inertia or voltage service, so nothing is paid twice.
* **EfW gate fees** (``efw_gate_fee_revenue_gbp``): WRAP's 2024/25 median EfW gate fee per tonne
  over Tolvik's 2024 export per tonne, per MWh dispatched (``efw_gate_fees_2050.json``). Paid by
  waste producers, not by the electricity bill.
* **Life-extension operating top-up** (``life_extension_top_up_gbp``): plant past its contract
  (expired offshore wind; unabated biomass, which holds no contract in 2050; expired Power
  BECCS) is life-extended at operating cost only (decisions 41 (a) and 42). Where its revenue,
  ancillary services included, does not cover its operating cost (fixed and variable O&M, fuel
  and carbon), the bill tops it up to break even on operating cost (never capital): the 2050
  bill's ``life_extension_operating_top_up`` line, which reads the register's private total
  (``cost_from`` the public account). Sized after the ancillary and gate-fee revenue.
* **Battery revenue top-up** (``battery_revenue_top_up_gbp``, decision 63, 3 Oct 2026): batteries
  stay short after their ancillary revenue, so the bill tops each battery row up to break even
  on its full required revenue (capital at the private rate included): the 2050 bill's
  ``battery_revenue_top_up`` line. Sized on the private case after every other income, the same
  payment per asset in every case (as the life-extension top-up and Sizewell C's RAB allowance):
  a state-held battery, financed at the lower rate, needs less, and the difference is the
  state's surplus, which the base incidence (bill-payers first) returns to bills. Battery rows
  only: long-duration stores (pumped hydro, LAES, CAES) are on the LDES cap and floor, and a
  carrier on both is refused.

* **Battery sub-hourly revenue** (``battery_subhourly_revenue_gbp``, decision 67, 3 Oct 2026): the
  hourly dispatch misses the intra-hour spreads batteries trade (Modo's GB benchmark; arXiv
  2510.01956 on 15-minute against hourly dispatch), so each battery row earns a declared uplift on
  its dispatched wholesale plus balancing revenue: +20% centrally, +40% the high case (reported as
  the battery top-up it would leave, ``battery_revenue_top_up_subhourly_high_gbp``, a memo, never
  income). Treatment: real market revenue the hourly bill lines do not carry (the intra-hour value
  of flexibility, paid by other market participants), so no bill line is added; it is the same
  per asset in every case, so it moves no case's resource saving, only who receives it (the state
  on its share; under the acquisition tariff, the state, credited to the tariff).

The amounts are asset properties, the same in every case. A balancing run group (the P2 bid
months) holds them at its market group's per-asset amounts (``--support-reference``,
:func:`support_from_reference`), so its bid stack moves no support payment the bill does not
carry (the dispatch step of the three-party identity).

Configured in ``public_ownership.support_lines`` (:func:`settings`): every item is off unless the
block names it. The FES25_HT_2050 entry turns them all on; a 2030 study may turn on the ancillary
services, the battery sub-hourly uplift and the battery top-up (the user, 3 Oct 2026), which work
on a non-vintaged register too (carriers map through :data:`PROVIDER_CARRIERS`, CP30 names
included; the 2030 table is ``ancillary_services_2030.csv``, the FY2024/25 outturn carried to 2030 on CP30's
drivers (user decision, 6 Oct 2026; ``ancillary_services_2030.py``), which the 2030 BSUoS lines hold exactly as
the 2050 lines hold the 2050 table). The life-extension
top-up stays with the 2050 stock-flow. With no block (every 2030 config today) nothing changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG_KEY = "support_lines"
#: The items ``public_ownership.support_lines`` may name.
ITEMS = ("ancillary_services", "efw_gate_fee", "life_extension_top_up", "battery_top_up", "battery_subhourly_uplift")
DEFAULT_BATTERY_CARRIERS = ("Battery",)
COLUMN_ANCILLARY = "ancillary_services_revenue_gbp"
COLUMN_ANCILLARY_VOLTAGE = "ancillary_voltage_stability_revenue_gbp"
COLUMN_GATE_FEE = "efw_gate_fee_revenue_gbp"
COLUMN_TOP_UP = "life_extension_top_up_gbp"
COLUMN_BATTERY_TOP_UP = "battery_revenue_top_up_gbp"
COLUMN_SUBHOURLY = "battery_subhourly_revenue_gbp"
COLUMN_BATTERY_TOP_UP_HIGH = "battery_revenue_top_up_subhourly_high_gbp"
#: F19 decision 5: the private-basis top-up each row would have been paid, kept beside the
#: owner-cost top-up when that basis is on (its presence marks the basis to later stages).
COLUMN_BATTERY_TOP_UP_PRIVATE_BASIS = "battery_revenue_top_up_private_basis_gbp"
#: F19 decision 5: the battery top-up's bases (``battery_top_up.basis``).
BATTERY_TOP_UP_OWNER_COST, BATTERY_TOP_UP_PRIVATE = "owner_cost", "private"
BATTERY_TOP_UP_BASES = (BATTERY_TOP_UP_OWNER_COST, BATTERY_TOP_UP_PRIVATE)
SUPPORT_COLUMNS = (COLUMN_ANCILLARY, COLUMN_GATE_FEE, COLUMN_SUBHOURLY, COLUMN_TOP_UP, COLUMN_BATTERY_TOP_UP)
#: Columns booked for the record only, never income: the voltage part of the ancillary revenue, the
#: battery top-up under the high sub-hourly case (decision 67's sensitivity) and the private-basis top-up.
MEMO_COLUMNS = (COLUMN_ANCILLARY_VOLTAGE, COLUMN_BATTERY_TOP_UP_HIGH, COLUMN_BATTERY_TOP_UP_PRIVATE_BASIS)
#: The bill lines the ancillary revenue is paid out of (``in_bill_line`` in the data file).
LINE_OTHER_SERVICES = "bsuos_other_services"
LINE_VOLTAGE = "bsuos_voltage_and_stability_constraints"
#: NESO's reserve trades, in the daily cost categories of the non-constraint BSUoS.
LINE_CATEGORIES = "balancing_categories"
#: Decision 65: each 2050 bill line and the data file's ``in_bill_line`` values its services part holds.
LINE_SERVICES_HELD = {LINE_OTHER_SERVICES: (LINE_OTHER_SERVICES, LINE_CATEGORIES), LINE_VOLTAGE: (LINE_VOLTAGE,)}
LINE_TOP_UP = "life_extension_operating_top_up"
LINE_BATTERY_TOP_UP = "battery_revenue_top_up"
#: Each top-up column and the bill line that pays it.
TOP_UP_LINES = {COLUMN_TOP_UP: LINE_TOP_UP, COLUMN_BATTERY_TOP_UP: LINE_BATTERY_TOP_UP}
#: Data-file provider -> the register carriers it pays (model plant only).
PROVIDER_CARRIERS = {
    "Battery": ("Battery", "battery"),
    "pumped_hydro": ("pumped_hydro", "Pumped Storage Hydroelectricity", "PHS"),
    "LAES": ("LAES",), "CAES": ("CAES",),
    "CCS Gas": ("CCS Gas",), "H2_turbine": ("H2_turbine",),
    # CP30 (2030) names: the FY2024/25 providers as the 2030 register carries them.
    "CCGT": ("CCGT",), "peaking_gas": ("OCGT", "gas_engine"), "CHP": ("CHP",),
    "nuclear": ("nuclear",), "large_hydro": ("large_hydro", "small_hydro", "hydro"),
    "biomass": ("biomass",), "CCS Biomass": ("CCS Biomass",),
    "wind_offshore": ("wind_offshore",), "wind_onshore": ("wind_onshore",),
    "waste_to_energy": ("waste_to_energy",),
}
#: Providers that are not register plant: their share stays in the bill line (consumers,
#: businesses, interconnector owners, NESO's synchronous-condenser contractors).
OUTSIDE_REGISTER = ("demand_side", "interconnector", "synchronous_condenser", "other")
CASES = ("central", "low", "high")
#: Top-up rules: the stock-flow technology and which rows of it are life-extended.
TOP_UP_RULES = ("expired", "all")
EXPIRED_BASIS = "contract_expired_merchant"


class SupportLinesError(ValueError):
    """A support-line setting or data file that cannot be used."""


def settings(raw: Mapping | None, root: Path | str = ROOT) -> dict | None:
    """``public_ownership.support_lines`` checked, with paths resolved (None when absent or all off).

    Each item is off unless the block names it; a named item is on unless ``enabled: false``.
    ``ancillary_services`` and ``efw_gate_fee`` take ``file`` (or ``table``) and ``case``;
    ``battery_top_up`` and ``battery_subhourly_uplift`` take ``carriers`` (default ``[Battery]``);
    the uplift takes ``share_by_case`` or ``central`` and ``high`` shares and the ``case`` in force.
    ``life_extension_top_up`` (the 2050 stock-flow's life-extended plant) works on a vintaged
    register only, which :func:`require_vintaged` enforces.
    """
    if not raw:
        return None
    raw = dict(raw)
    unknown = sorted(set(raw) - set(ITEMS))
    if unknown:
        raise SupportLinesError(f"{CONFIG_KEY} has unknown keys {unknown}; supported: {list(ITEMS)}")
    blocks = {}
    for key in ITEMS:
        block = raw.get(key)
        if block is None:
            continue
        block = dict(block or {})
        if not bool(block.pop("enabled", True)):
            continue
        blocks[key] = block
    out = {}
    for key in ("ancillary_services", "efw_gate_fee"):
        if key not in blocks:
            continue
        block = blocks[key]
        case = str(block.get("case", "central"))
        if case not in CASES:
            raise SupportLinesError(f"{CONFIG_KEY}.{key}.case={case!r}; one of {CASES}")
        if block.get("file") and block.get("table"):
            # Review 2: a study's block deep-merges over the defaults' (2030 names ``table``), so both would
            # be ambiguous; name one.
            raise SupportLinesError(f"{CONFIG_KEY}.{key} names both file and table; name one")
        file = block.get("file", block.get("table"))
        if not file:
            raise SupportLinesError(f"{CONFIG_KEY}.{key} needs file (or table)")
        path = Path(file)
        out[key] = {"file": path if path.is_absolute() else Path(root) / path, "case": case}
    if "life_extension_top_up" in blocks:
        rules = {str(k): str(v) for k, v in dict(blocks["life_extension_top_up"].get("technologies") or {}).items()}
        if bad := sorted(t for t, r in rules.items() if r not in TOP_UP_RULES):
            raise SupportLinesError(f"{CONFIG_KEY}.life_extension_top_up.technologies {bad}: one of {TOP_UP_RULES}")
        out["life_extension_top_up"] = {"technologies": rules}
    if "battery_top_up" in blocks:
        carriers = [str(c) for c in (blocks["battery_top_up"].get("carriers") or DEFAULT_BATTERY_CARRIERS)]
        # F19 decision 5: each case's top-up on its owner's required revenue (owner_cost, the code
        # default and config), or the private case's paid to every owner (private, as decided 3 Oct).
        basis = str(blocks["battery_top_up"].get("basis", BATTERY_TOP_UP_OWNER_COST))
        if basis not in BATTERY_TOP_UP_BASES:
            raise SupportLinesError(f"{CONFIG_KEY}.battery_top_up.basis={basis!r}; one of {BATTERY_TOP_UP_BASES}")
        in_vintaged = blocks["battery_top_up"].get("owner_cost_in_vintaged_registers", False)
        if not isinstance(in_vintaged, bool):
            raise SupportLinesError(f"{CONFIG_KEY}.battery_top_up.owner_cost_in_vintaged_registers must be true or false")
        out["battery_top_up"] = {"carriers": carriers, "basis": basis, "owner_cost_in_vintaged_registers": in_vintaged}
    if "battery_subhourly_uplift" in blocks:
        uplift = blocks["battery_subhourly_uplift"]
        carriers = [str(c) for c in (uplift.get("carriers") or DEFAULT_BATTERY_CARRIERS)]
        if uplift.get("share_by_case") and any(k in uplift for k in ("central", "high")):
            raise SupportLinesError(f"{CONFIG_KEY}.battery_subhourly_uplift names both share_by_case and "
                                    "central/high; name one form")
        shares = {str(k): float(v) for k, v in dict(uplift.get("share_by_case") or {}).items()}
        shares.update({k: float(uplift[k]) for k in ("central", "high") if k in uplift})
        case = str(uplift.get("case", "central"))
        if case not in shares or "high" not in shares or any(not 0.0 <= v < 1.0 for v in shares.values()):
            raise SupportLinesError(f"{CONFIG_KEY}.battery_subhourly_uplift needs central and high shares "
                                    f"(each in [0, 1)) holding its case {case!r}")
        out["battery_subhourly_uplift"] = {"carriers": carriers, "share": shares[case], "high_share": shares["high"],
                                           "case": case}
    return out or None


def require_vintaged(support: Mapping | None, vintaged: bool) -> None:
    """The life-extension top-up reads the 2050 stock-flow's technologies: refuse it on any other register."""
    if support and support.get("life_extension_top_up") and not vintaged:
        raise SupportLinesError(f"{CONFIG_KEY}.life_extension_top_up applies to a vintaged (2040 on) register only")

# ── ancillary services ───────────────────────────────────────────────────────

def load_ancillary(path) -> pd.DataFrame:
    """The ancillary-services table, checked: shares sum to one per service, providers known.

    The spend columns are ``spend_2050_<case>_gbp_m_2024`` (the 2050 table) or
    ``spend_<case>_gbp_m_2024`` (a table for another year, ``ancillary_services_2030.csv``);
    the second is read as the first.
    """
    table = pd.read_csv(path)
    table = table.rename(columns={f"spend_{c}_gbp_m_2024": f"spend_2050_{c}_gbp_m_2024" for c in CASES
                                  if f"spend_2050_{c}_gbp_m_2024" not in table.columns})
    need = {"service", "provider_carrier", "participation_share", "in_bill_line",
            "spend_2050_central_gbp_m_2024", "spend_2050_low_gbp_m_2024", "spend_2050_high_gbp_m_2024"}
    if missing := need - set(table.columns):
        raise SupportLinesError(f"{path} lacks {sorted(missing)}")
    unknown = sorted(set(table["provider_carrier"]) - set(PROVIDER_CARRIERS) - set(OUTSIDE_REGISTER))
    if unknown:
        raise SupportLinesError(f"{path}: unknown providers {unknown}")
    sums = table.groupby("service")["participation_share"].sum()
    if bad := sorted(sums.index[(sums - 1.0).abs() > 1e-3]):
        raise SupportLinesError(f"{path}: participation shares do not sum to 1 for {bad}")
    return table


def ancillary_by_provider(table: pd.DataFrame, case: str = "central") -> pd.DataFrame:
    """GBP (2024 money) each provider earns by bill line: service spend x participation share."""
    spend = pd.to_numeric(table[f"spend_2050_{case}_gbp_m_2024"], errors="coerce").fillna(0.0) * 1e6
    frame = table.assign(gbp=spend * pd.to_numeric(table["participation_share"], errors="coerce").fillna(0.0))
    return frame.groupby(["provider_carrier", "in_bill_line"], as_index=False)["gbp"].sum()


def ancillary_by_row(assets: pd.DataFrame, table: pd.DataFrame, case: str = "central") -> tuple[pd.DataFrame, dict]:
    """Each register row's ancillary revenue (total and its voltage-and-stability part), and the allocation report.

    A provider's money goes to its register rows pro rata to capacity (one row per asset or
    cohort, the private case's rows). A provider with no register capacity keeps its money in
    the bill line (reported as unallocated), as the outside-register providers do.
    """
    by = ancillary_by_provider(table, case)
    carrier = assets["carrier"].astype(str)
    capacity = pd.to_numeric(assets["capacity_mw"], errors="coerce").fillna(0.0)
    total = np.zeros(len(assets))
    voltage = np.zeros(len(assets))
    report = {"case": case, "by_provider_gbp": {}, "unallocated_gbp": {}, "rate_gbp_per_kw_year": {}}
    for provider, rows in by.groupby("provider_carrier"):
        money = float(rows["gbp"].sum())
        in_voltage = float(rows.loc[rows["in_bill_line"].eq(LINE_VOLTAGE), "gbp"].sum())
        report["by_provider_gbp"][provider] = money
        if provider in OUTSIDE_REGISTER:
            report["unallocated_gbp"][provider] = money
            continue
        mask = carrier.isin(PROVIDER_CARRIERS[provider]).to_numpy()
        mw = float(capacity[mask].sum())
        if mw <= 0:
            report["unallocated_gbp"][provider] = money
            continue
        share = capacity.to_numpy() * mask / mw
        total += money * share
        voltage += in_voltage * share
        report["rate_gbp_per_kw_year"][provider] = money / (mw * 1e3)
    by_line = by.groupby("in_bill_line")["gbp"].sum()
    report["by_bill_line_gbp"] = {str(k): float(v) for k, v in by_line.items()}
    report["services_held_by_bill_line_gbp"] = services_held_by_line(table, case)
    report["allocated_gbp"] = float(total.sum())
    out = pd.DataFrame({COLUMN_ANCILLARY: total, COLUMN_ANCILLARY_VOLTAGE: voltage}, index=assets["asset_id"].astype(str))
    return out, report


def services_held_by_line(table: pd.DataFrame, case: str = "central") -> dict[str, float]:
    """Decision 65: GBP (2024 money) of services each 2050 BSUoS line holds, every provider included.

    The sum of each service's spend over the data file's ``in_bill_line`` values a line holds
    (:data:`LINE_SERVICES_HELD`): exactly what :func:`ancillary_by_provider` pays out of it.
    """
    if case not in CASES:
        raise SupportLinesError(f"ancillary case {case!r}; one of {CASES}")
    by = ancillary_by_provider(table, case)
    unknown = sorted(set(by["in_bill_line"]) - {v for values in LINE_SERVICES_HELD.values() for v in values})
    if unknown:
        raise SupportLinesError(f"ancillary services paid out of lines no bill line holds: {unknown}")
    return {line: float(by.loc[by["in_bill_line"].isin(parts), "gbp"].sum())
            for line, parts in LINE_SERVICES_HELD.items()}


# ── EfW gate fees ────────────────────────────────────────────────────────────

def gate_fee_gbp_per_mwh(path, case: str = "central") -> float:
    """The gate fee per MWh exported (GBP 2024) from ``efw_gate_fees_2050.json``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    value = float(data["gate_fee_gbp_per_mwh_2024"][case])
    if not np.isfinite(value) or value < 0:
        raise SupportLinesError(f"{path}: gate fee {value}")
    return value


def gate_fee_by_row(assets: pd.DataFrame, fee_gbp_per_mwh: float) -> pd.Series:
    """Gate-fee revenue per row: the fee x the row's dispatch, on energy-from-waste rows (``waste_to_energy``)."""
    efw = assets["carrier"].astype(str).eq("waste_to_energy")
    dispatch = pd.to_numeric(assets["dispatch_mwh"], errors="coerce").fillna(0.0)
    return pd.Series(np.where(efw, dispatch * float(fee_gbp_per_mwh), 0.0), index=assets["asset_id"].astype(str))


# ── life-extension operating top-up ──────────────────────────────────────────

def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def top_up_targets(rows: pd.DataFrame, rules: Mapping[str, str]) -> pd.Series:
    """The rows the top-up covers: a technology's every row (``all``) or its expired-contract rows (``expired``)."""
    technology = rows["stock_flow_technology"].fillna("").astype(str) if "stock_flow_technology" in rows.columns \
        else pd.Series("", index=rows.index)
    basis = rows["support_basis"].fillna("").astype(str) if "support_basis" in rows.columns \
        else pd.Series("", index=rows.index)
    out = pd.Series(False, index=rows.index)
    for name, rule in rules.items():
        hit = technology.eq(name)
        out |= hit if rule == "all" else hit & basis.eq(EXPIRED_BASIS)
    return out


def operating_shortfall(rows: pd.DataFrame) -> pd.Series:
    """Operating cost (fixed and variable O&M, fuel and carbon) less revenue, every income included (GBP)."""
    revenue = (_num(rows, "market_revenue_gbp") + _num(rows, "support_revenue_gbp")
               + _num(rows, "capacity_market_revenue_gbp") + _num(rows, "carbon_removal_cfd_revenue_gbp")
               + _num(rows, COLUMN_ANCILLARY) + _num(rows, COLUMN_GATE_FEE))
    cost = _num(rows, "annual_fixed_om_gbp") + _num(rows, "annual_variable_om_gbp") + _num(rows, "fuel_carbon_gbp")
    return cost - revenue


def top_up_by_row(private_rows: pd.DataFrame, rules: Mapping[str, str]) -> pd.Series:
    """The top-up each target row is paid: its operating shortfall, floored at zero (never capital).

    Measured on the private case's rows (a payment to the plant whoever owns it); indexed by asset id.
    """
    targets = top_up_targets(private_rows, rules)
    shortfall = operating_shortfall(private_rows).clip(lower=0.0)
    return pd.Series(np.where(targets, shortfall, 0.0), index=private_rows["asset_id"].astype(str))


def subhourly_by_row(assets: pd.DataFrame, carriers, share: float) -> pd.Series:
    """Decision 67: ``share`` of each battery row's dispatched wholesale plus balancing revenue (floored at zero)."""
    target = assets["carrier"].astype(str).isin([str(c) for c in carriers])
    revenue = (_num(assets, "wholesale_revenue_gbp") + _num(assets, "balancing_settlement_gbp")).clip(lower=0.0)
    return pd.Series(np.where(target, float(share) * revenue, 0.0), index=assets["asset_id"].astype(str))


def battery_top_up_by_row(private_rows: pd.DataFrame, carriers, ldes_carriers=()) -> pd.Series:
    """Decision 63: each battery row's shortfall on its full required revenue, after every other income.

    ``private_rows`` are the private case's rows with every other support line already booked
    (their ``producer_rent_gbp`` is revenue, Capacity Market and ancillary services included,
    less fuel and the required revenue at the private rate). The top-up is the negative rent,
    floored at zero; rows without a rent (costs unresolved) get none. Indexed by asset id.
    """
    carriers = [str(c) for c in carriers]
    if overlap := sorted(set(carriers) & {str(c) for c in ldes_carriers}):
        raise SupportLinesError(f"battery_top_up carriers {overlap} are on the LDES cap and floor too; "
                                "a store is supported by one or the other")
    target = private_rows["carrier"].astype(str).isin(carriers)
    rent = pd.to_numeric(private_rows["producer_rent_gbp"], errors="coerce") \
        if "producer_rent_gbp" in private_rows.columns else pd.Series(np.nan, index=private_rows.index)
    shortfall = (-rent).clip(lower=0.0).fillna(0.0)
    return pd.Series(np.where(target, shortfall, 0.0), index=private_rows["asset_id"].astype(str))


def owner_cost_top_up(support: Mapping | None, vintaged: bool) -> list[str] | None:
    """F19 decision 5: the battery carriers topped up at each owner's cost, or None (the private basis).

    On only where ``battery_top_up.basis`` is ``owner_cost``, and in a vintaged (2040 on) register
    only when ``owner_cost_in_vintaged_registers`` says so (2050 keeps the private basis).
    """
    block = (support or {}).get("battery_top_up")
    if not block or block.get("basis", BATTERY_TOP_UP_OWNER_COST) != BATTERY_TOP_UP_OWNER_COST:
        return None
    if vintaged and not block.get("owner_cost_in_vintaged_registers", False):
        return None
    return [str(c) for c in block.get("carriers") or DEFAULT_BATTERY_CARRIERS]


def rebook_battery_top_up_at_owner_cost(register: pd.DataFrame, carriers) -> pd.DataFrame:
    """F19 decision 5: each case's battery top-up on its own row's required revenue.

    :func:`book_support` books the private case's shortfall to every case's row (a payment to
    the plant whoever owns it).  Here each row is re-sized on its own case: the top-up is the
    negative of its rent before the top-up, floored at zero, so a state row (cheaper capital)
    or a joint venture's (the stake's rate) is paid only its own shortfall, and the private
    row is unchanged.  The rent and the state's surplus move by the difference (its owners'
    shares of it); the private-basis amount is kept in
    :data:`COLUMN_BATTERY_TOP_UP_PRIVATE_BASIS`.  Rows without a rent are left alone.
    """
    if register.empty or COLUMN_BATTERY_TOP_UP not in register.columns:
        return register
    register = register.copy()
    paid = pd.to_numeric(register[COLUMN_BATTERY_TOP_UP], errors="coerce").fillna(0.0)
    register[COLUMN_BATTERY_TOP_UP_PRIVATE_BASIS] = paid
    rent = pd.to_numeric(register["producer_rent_gbp"], errors="coerce") if "producer_rent_gbp" in register.columns \
        else pd.Series(np.nan, index=register.index)
    target = register["carrier"].astype(str).isin([str(c) for c in carriers]) & rent.notna()
    if not target.any():
        return register
    own = (-(rent - paid)).clip(lower=0.0)
    delta = (own - paid).where(target, 0.0)
    register.loc[target, COLUMN_BATTERY_TOP_UP] = own[target]
    register.loc[target, "producer_rent_gbp"] = rent[target] + delta[target]
    if "public_surplus_before_recycling_gbp" in register.columns:
        share = pd.to_numeric(register.get("ownership_share", 0.0), errors="coerce").fillna(0.0)
        surplus = pd.to_numeric(register["public_surplus_before_recycling_gbp"], errors="coerce")
        register.loc[target, "public_surplus_before_recycling_gbp"] = surplus[target] + share[target] * delta[target]
    return register


def support_from_reference(reference: pd.DataFrame, columns, assets: pd.DataFrame) -> dict:
    """The support lines a balancing run group holds at its market group's per-asset amounts.

    ``reference`` is the market group's private rows (``asset_id`` and the support columns);
    ``columns`` the support columns this group configures. Refuses an asset the reference does
    not cover, or a configured column it lacks.
    """
    frame = reference.assign(asset_id=reference["asset_id"].astype(str)).drop_duplicates("asset_id")
    frame = frame.set_index("asset_id")
    if missing := sorted(set(columns) - set(frame.columns)):
        raise SupportLinesError(f"the support reference lacks {missing}; rebuild its register first")
    ids = assets["asset_id"].astype(str).drop_duplicates()
    if absent := sorted(set(ids) - set(frame.index))[:5]:
        raise SupportLinesError(f"the support reference has no row for assets such as {absent}")
    return {c: pd.to_numeric(frame.loc[ids, c], errors="coerce").fillna(0.0).astype(float) for c in columns}


# ── booking ──────────────────────────────────────────────────────────────────

def book_support(register: pd.DataFrame, by_asset: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Book each support line to every case's rows: its column, the producer rent and the state's share of it.

    ``by_asset`` maps a column of :data:`SUPPORT_COLUMNS` (or the voltage part) to GBP per asset
    id. The amount is the asset's whoever owns it; ``ownership_share`` gives the state its part.
    """
    if not by_asset or register.empty:
        return register
    register = register.copy()
    ids = register["asset_id"].astype(str)
    has_rent = register["producer_rent_gbp"].notna() if "producer_rent_gbp" in register.columns \
        else pd.Series(False, index=register.index)
    share = pd.to_numeric(register.get("ownership_share", 0.0), errors="coerce").fillna(0.0)
    for column, series in by_asset.items():
        paid = ids.map(series).fillna(0.0).astype(float)
        register[column] = paid
        if column in MEMO_COLUMNS:
            continue          # the voltage part of the ancillary column, or a sensitivity memo: never income
        register.loc[has_rent, "producer_rent_gbp"] = (pd.to_numeric(register.loc[has_rent, "producer_rent_gbp"],
                                                                     errors="coerce") + paid[has_rent])
        if "public_surplus_before_recycling_gbp" in register.columns:
            register.loc[has_rent, "public_surplus_before_recycling_gbp"] = (
                pd.to_numeric(register.loc[has_rent, "public_surplus_before_recycling_gbp"], errors="coerce")
                + share[has_rent] * paid[has_rent])
    return register


def support_report(private_rows: pd.DataFrame, ancillary_report: Mapping | None = None) -> dict:
    """By stock-flow technology: each support line, the producer rent after them, and whether the private case breaks even."""
    rows = private_rows.copy()
    technology = rows["stock_flow_technology"].fillna(rows["carrier"].astype(str)).astype(str) \
        if "stock_flow_technology" in rows.columns else rows["carrier"].astype(str)
    frame = pd.DataFrame({"technology": technology, "rent": _num(rows, "producer_rent_gbp"),
                          **{c: _num(rows, c) for c in SUPPORT_COLUMNS}})
    grouped = frame.groupby("technology").sum()
    out = {t: {"producer_rent_after_support_gbp": float(r["rent"]),
               **{c: float(r[c]) for c in SUPPORT_COLUMNS},
               "breaks_even": bool(r["rent"] >= -1.0)} for t, r in grouped.iterrows()
           if any(abs(float(r[c])) > 0 for c in SUPPORT_COLUMNS) or t in ("Lithium batteries", "Energy from waste")}
    return {"by_technology": out, "ancillary_allocation": dict(ancillary_report or {})}
