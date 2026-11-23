"""Generate notebooks/fes_ht_2050_results.ipynb: the 2050 public-ownership results (FES 2050 plan, Part 5).

Run from anywhere:  python scripts/analysis/build_fes2050_results_notebook.py
then execute:       jupyter nbconvert --to notebook --execute --inplace notebooks/fes_ht_2050_results.ipynb

Rule ``build_fes2050_results_notebook`` (rules/fes2050.smk) does both; it is not part of ``all``.

The notebook reads ONLY output files: the fes_ht_2050 bill, balancing, tariff, institution,
public-account and ledger outputs, the 2030-2050 pathway, the FES25_HT_2050 solved months'
constraint-cost tables, the gate weeks of decision 29, the Y8 validation outputs, the register
bounds' ledgers (decisions 11, 62 and 68), and, for the side-by-side, the cp30_2030_cp30_network ledger and
bill components. It quotes no number of its own. Every title carries a "provisional" tag until
the decision-49 rebuild and every solve and finance output after it are on disk, and the resolved
finance config carries decisions 58, 63, 66, 68 and 69 (the provenance cell says what breaks the
chain).

Decisions 53 to 69 (3 Oct 2026) added: the risk-adjusted (Lucas) columns of the ledger
(``resource_saving_risk_adjusted_gbp``, ``taxpayer_risk_bearing_gbp``,
``exchequer_change_risk_adjusted_gbp``), the co-investors' implicit guarantee
(``co_investment_implicit_guarantee_gbp``), the support lines (public-account
``*_revenue_gbp`` / ``*_top_up_gbp`` and the bill lines ``battery_revenue_top_up`` and
``life_extension_operating_top_up``), the register bounds, the bill sensitivities
(``levy_exemption_electrolysers_2050``, ``<line>_ancillary_low/high``,
``battery_revenue_top_up_subhourly_high``) and the P3 tariff's ``diagnostics.decision_55``. Every
one is optional: a section whose output is missing says so and is skipped.

Set PYPSA_GB_ROOT to execute a copy of the notebook outside the repository.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "fes_ht_2050_results.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =====================================================================================================
md(r"""
# FES 2050 results: the household bill and what public ownership does to it

Calendar 2050 on FES 2025 Holistic Transition, 2013 weather: twelve monthly market runs (Stage 1
day-ahead plus the balancing mechanism), twelve central-dispatch months, and the P2 bid-stack months at
each deployment share. Bills are GBP per year for a 3,100 kWh benchmark household, 2025 real prices;
system figures are GBP bn a year, 2024 real (the finance layer's money).

1. **The 2050 bill by case**: every bill line, private against P1, P2 and its deployment shares (state bids), and P3 under both institutions.
2. **Who gains: the three-party columns**: consumers, the Exchequer and private investors, summing to the resource saving.
   - **2a. The risk-adjusted (Lucas) view** (decision 69): the cost-of-capital saving read as the price of risk taxpayers bear, beside the headline.
   - **2b. Support for the private counterfactual** (decisions 58, 63, 65 and 67): the ancillary services, gate fees, sub-hourly revenue and the two top-up lines.
3. **Sensitivities**: the public rate at the PWLB standard margin (decision 54; +1 pp before F29) and the NESO wind ramp, the electrolyser levy exemption (decision 60) and the bill sensitivities; the register bounds (repowering, co-ownership, learning, retirement); HPC and Sizewell C under P3 (decision 55).
4. **2030 against 2050**: the same cases in both snapshot years, and why the saving changes source.
5. **The pathway 2030-2050**: the annual benefit by mechanism, the totals and NPVs, and the beyond-2050 line.
6. **The balancing mechanism by month**: load shed and constraint cost, market against central dispatch.
7. **European prices and validation**: the TYNDP Global Ambition and NT+ sensitivities, and the dispatch checks against FES ES1.

**Read the provenance box first.** Until the decision-49 rebuild and re-solve are on disk, and the finance config
carries decisions 58 to 69, every title is tagged provisional. The notebook carries no numbers of its own: re-run it
and every figure and table refreshes.
""")

code(r"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PYPSA_GB_ROOT", Path.cwd()))
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
RES = ROOT / "resources"
PO, AN, MK, NET = RES / "public_ownership", RES / "analysis", RES / "market", RES / "network"
GROUP, STUDY = "fes_ht_2050", "FES25_HT_2050"
GROUP_2030, CASE_2030 = "cp30_2030_cp30_network", "CP30_2030_cp30_network"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
SHARES = [25, 50, 75, 100]
# Decision 49 (3 Oct 2026): one rebuild of FES25_HT_2050 and one re-solve of every 2050 month. Outputs from
# before the rebuild are provisional. The rebuild was prepared on the evening of 3 October and launched later.
DECISION_49_PREPARED = datetime(2026, 10, 3, 18, 0)

P1, P2 = "public_control", "public_ownership"
P3_I1 = "public_ownership@fully_nationalised_market_value"
P3_I3 = "public_ownership@fully_nationalised_market_value_central_dispatch"
BIDS = {s: f"public_ownership@deployment_share_{s}_state_bids" for s in SHARES}
CASES = [("private", "Private", "market"), (P1, "P1", "market"), (P2, "P2\nmarket bids", "market")]
CASES += [(BIDS[s], f"P2 {s}%\nstate bids", "bids") for s in SHARES]
CASES += [(P3_I1, "P3 I1\nmarket", "tariff"), (P3_I3, "P3 I3\ncentral", "institution")]
LABEL = {c: l for c, l, _ in CASES}
flat = lambda c: LABEL[c].replace("\n", " ")

# ---- style (the 2030 notebook's) ---------------------------------------------------------------
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#84837c", "#e6e5e1"
# Reference categorical palette in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SAVING, COST, NEUTRAL = "#1baf7a", "#e34948", "#b9b7b0"
mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 115,
})
gbp = lambda v, d=0: f"£{v:,.{d}f}"
fmt_gbp = mpl.ticker.StrMethodFormatter("£{x:,.0f}")
fmt_bn = mpl.ticker.StrMethodFormatter("£{x:,.0f}bn")
TAG = ""   # set by the provenance cell


def title(ax, text, sub=None, tagged=True):
    ax.set_title(text + (TAG if tagged else ""), pad=20 if sub else 8, color=INK)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    fontsize=9, color=INK_3, va="bottom", ha="left", annotation_clip=False)


def caption(fig, text, y=-0.01):
    fig.text(0.01, y, "\n".join(textwrap.wrap(text, 190)), fontsize=8.5, color=INK_3, va="top")


MISSING: list[str] = []


def have(*paths) -> bool:
    gone = [str(Path(p).relative_to(ROOT)) for p in paths if not Path(p).exists()]
    MISSING.extend(g for g in gone if g not in MISSING)
    return not gone


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mtime(path):
    path = Path(path)
    return datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else None


def box(lines, width=112):
    wrapped = []
    for x in lines:
        wrapped += textwrap.wrap(x, width, subsequent_indent=" " * (len(x) - len(x.lstrip()) + 4)) if len(x) > width else [x]
    w = max(map(len, wrapped)) + 2
    print("+" + "-" * w + "+")
    for x in wrapped:
        print("| " + x.ljust(w - 1) + "|")
    print("+" + "-" * w + "+")
""")

# =====================================================================================================
md(r"""
## Provenance

The results are final only when (a) the FES25_HT_2050 network has been rebuilt after decision 49 was prepared,
(b) every market, central and P2 bid-stack month has been solved on that network, (c) every finance output
the notebook reads was written after the last of those solves, and (d) the resolved finance config carries the
3 October decisions: the merchant level annuity (66), the support lines (58, 63), separate co-ownership finance (68)
and the risk-adjusted view (69). Anything else is provisional, and the box lists what breaks the chain. The register
bounds are sensitivities: the box lists them, but they do not decide the status.
""")

code(r"""
F = {  # the outputs this notebook reads, by role
    "bill by case (market)": PO / f"{GROUP}_consumer_bill_by_ownership_case.csv",
    "bill by case summary": PO / f"{GROUP}_consumer_bill_by_ownership_case_summary.json",
    "P2 state-bid bills": PO / f"{GROUP}_consumer_bill_balancing_cases.csv",
    "P2 state-bid summary": PO / f"{GROUP}_consumer_bill_balancing_cases_summary.json",
    "P3 tariff lines (I1)": PO / f"{GROUP}_nationalised_bill_components.csv",
    "P3 institution (I3)": PO / f"{GROUP}_nationalised_bill_by_institution.csv",
    "P3 institution summary": PO / f"{GROUP}_nationalised_bill_by_institution.json",
    "three-party ledger": PO / f"{GROUP}_exchequer_ledger.csv",
    "ledger summary": PO / f"{GROUP}_exchequer_ledger.json",
    "public account summary": PO / f"{GROUP}_public_account_summary.json",
    "P3 tariff summary": PO / f"{GROUP}_nationalised_bill_summary.json",
    "private bill summary": AN / f"{STUDY}_consumer_bill_future_summary.json",
    "pathway 2030-2050": PO / "pathway_2030_2050.csv",
    "pathway summary": PO / "pathway_2030_2050_summary.json",
    "private bill lines": AN / f"{STUDY}_consumer_bill_future_components.csv",
    "market BM costs (bill basis)": AN / f"{STUDY}_future_balancing_costs_monthly.csv",
    "ES1 dispatch validation": AN / f"{STUDY}_fes_energy_validation.csv",
    "ES1 validation summary": AN / f"{STUDY}_fes_energy_validation.json",
    "IC1 diagnostics": AN / f"{STUDY}_ic1_diagnostics_summary.json",
    "BM storage churn": AN / f"{STUDY}_bm_storage_churn.json",
    "2030 ledger": PO / f"{GROUP_2030}_exchequer_ledger.csv",
    "2030 private bill lines": AN / f"{CASE_2030}_consumer_bill_future_components.csv",
}
FINANCE_2050 = [k for k in F if not k.startswith(("2030", "ES1", "IC1", "BM storage"))]
# The register bounds (no re-solve; each re-accounts the central months with one group-level block changed).
# Sensitivities: listed in the box, never part of the final test.
BOUNDS = {"Life extension, onshore repowering share 0.5 (decision 62)": "fes_ht_2050_repowering_bound",
          "Co-owner halo, 0.5 bp per point (decision 68)": "fes_ht_2050_coownership_reduction_bound",
          "Pooled co-ownership, leakage test (decision 68)": "fes_ht_2050_coownership_pooled_bound",
          "CCC learning rate (decision 11)": "fes_ht_2050_learning_bound",
          "Private life extension, no rebuild (F28 toggle; the central before 27 Oct 2026)": "fes_ht_2050_retirement_bound",
          "Auction margin 0 (central 5%; decision 71)": "fes_ht_2050_auction_margin_none_bound",
          "Auction margin 10% (central 5%; decision 71)": "fes_ht_2050_auction_margin_high_bound"}
BOUND_LEDGERS = {name: PO / f"{g}_exchequer_ledger.csv" for name, g in BOUNDS.items()}
FINANCE_CONFIG = PO / f"{STUDY}_finance_config.yaml"


def finance_config_checks(path):
    # Decisions 58, 63, 66, 68 and 69 (3 Oct 2026) in the resolved study config the finance scripts read.
    try:
        import yaml
        po = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("public_ownership") or {}
    except Exception as exc:          # missing file or no yaml: every check fails, and the box says why
        return {f"finance config unreadable ({type(exc).__name__})": False}
    support = po.get("support_lines") or {}
    return {
        "66 merchant level annuity": (po.get("vintage_capital_2050") or {}).get("merchant_capital") == "level_annuity",
        "58/63 support lines": all(k in support for k in ("ancillary_services", "efw_gate_fee", "life_extension_top_up",
                                                         "battery_top_up", "battery_subhourly_uplift")),
        "68 separate co-ownership finance": (po.get("co_ownership_finance") or {}).get("basis") == "separate",
        "69 risk-adjusted view": bool((po.get("risk_adjusted_view") or {}).get("enabled")),
    }


CONFIG_CHECKS = finance_config_checks(FINANCE_CONFIG)
network = NET / f"{STUDY}.nc"
solved = {f"{m} market": MK / f"{STUDY}_{m}_market_balancing.nc" for m in MONTHS}
solved |= {f"{m} central": MK / f"{STUDY}_{m}_central_dispatch_balancing.nc" for m in MONTHS}
solved |= {f"{m} P2 {s}%": MK / f"{STUDY}_{m}_p2_bids_{s}pct_balancing.nc" for m in MONTHS for s in SHARES}

net_t = mtime(network)
solved_t = {k: mtime(p) for k, p in solved.items()}
unsolved = [k for k, t in solved_t.items() if t is None]
before_net = [k for k, t in solved_t.items() if t is not None and net_t is not None and t < net_t]
last_solve = max((t for t in solved_t.values() if t), default=None)
stale = [k for k in FINANCE_2050 if mtime(F[k]) is None or (last_solve and mtime(F[k]) < last_solve)]
rebuilt = net_t is not None and net_t > DECISION_49_PREPARED
config_ok = all(CONFIG_CHECKS.values())
IS_FINAL = rebuilt and not unsolved and not before_net and not stale and config_ok
TAG = "" if IS_FINAL else "   [provisional]"

lines = [f"STATUS        {'FINAL: decision-49 rebuild, re-solve and finance chain all on disk' if IS_FINAL else 'PROVISIONAL, refreshed after decision 49'}",
         f"NETWORK       {network.relative_to(ROOT)}  written {net_t:%Y-%m-%d %H:%M}" if net_t else "NETWORK       not found",
         f"              {'rebuilt after' if rebuilt else 'NOT rebuilt since'} decision 49 was prepared ({DECISION_49_PREPARED:%d %b %H:%M})",
         f"SOLVED MONTHS {len(solved) - len(unsolved)}/{len(solved)} on disk; {len(before_net)} older than the network; last solve "
         + (f"{last_solve:%Y-%m-%d %H:%M}" if last_solve else "none"),
         f"FINANCE       {len(FINANCE_2050) - len(stale)}/{len(FINANCE_2050)} outputs newer than the last solve"
         + (f"; stale or missing: {', '.join(stale)}" if stale else ""),
         f"CONFIG        {FINANCE_CONFIG.relative_to(ROOT)}: "
         + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in CONFIG_CHECKS.items()),
         "", "SOURCES"]
for k, p in F.items():
    t = mtime(p)
    lines.append(f"  {t:%Y-%m-%d %H:%M}  {k:<28} {p.relative_to(ROOT)}" if t else f"  MISSING           {k:<28} {p.relative_to(ROOT)}")
lines += ["", "REGISTER BOUNDS (sensitivities; not part of the status)"]
central_ledger_t = mtime(F["three-party ledger"])
for k, p in BOUND_LEDGERS.items():
    t = mtime(p)
    note = "" if not t or not central_ledger_t or t >= central_ledger_t else "  (older than the central ledger)"
    lines.append(f"  {t:%Y-%m-%d %H:%M}  {k:<48} {p.relative_to(ROOT)}{note}" if t
                 else f"  MISSING           {k:<48} {p.relative_to(ROOT)}")
if unsolved:
    lines += ["", "Not solved: " + ", ".join(unsolved[:12]) + (" ..." if len(unsolved) > 12 else "")]
lines += ["", "PRICE BASIS   bills GBP real 2025 per 3,100 kWh (VAT 5% once); system GBP real 2024; NPVs to 2030",
          "The 2030 snapshot is CP30 Further Flex and Renewables, not FES HT 2030, on the 2030 cost table (decision 26: the 2050",
          "vintage table is on the full DESNZ basis, with grid-connection capex and insurance). State the basis beside any comparison."]
box(lines, width=140)
""")

# =====================================================================================================
md(r"""
## 1. The 2050 bill by case

Every bill line, grouped. The cases are the private counterfactual, P1 (a 30% state stake in new build), P2 with
market bids (the state owns all post-2025 new build), P2 at 25, 50, 75 and 100% deployment shares with the state's
own balancing bids at cost (decision 46: these are the P2 headline), and P3 (the state also buys the existing fleet
in 2030) under market machinery (I1) and central dispatch (I3).

Under P3 the market-construct lines (wholesale, CfD, DPA, constraint payments, Capacity Market) are replaced by the
cost of generation and storage, so its stack has a different shape: compare the totals, not the bands.
""")

code(r"""
VAT = 0.05
CONTRACTS = {"cfd", "dpa", "allocated_ccs", "remaining_ccs", "sizewell_c_rab", "ldes_cap_and_floor",
             "nuclear_contract_counterparty"}
OWNERSHIP_LINES = {"new_build_contracts_at_cost", "pay_as_bid_ceiling", "sizewell_c_rab_public_rate_adjustment",
                   "beccs_carbon_removal_levy"}
GROUPS = ["Wholesale and hedging", "Balancing, system and losses", "Contracts: CfD, DPA, CCS, RAB, LDES", "Networks",
          "Supplier costs and margin", "Other policy levies", "Cost of generation and storage (nationalised)",
          "Ownership adjustments"]
GROUP_COLOUR = dict(zip(GROUPS, SERIES))
CATEGORY_GROUP = {"Wholesale & hedging": GROUPS[0], "System operation, balancing & losses": GROUPS[1], "Network costs": GROUPS[3],
                  "Supplier, payment, margin & headroom": GROUPS[4], "Policy costs": GROUPS[5]}


def group_of(cid, category):
    if cid in CONTRACTS:
        return GROUPS[2]
    if (cid.startswith(("public_", "legacy_levy_")) or cid.endswith("_public_financing_adjustment") or cid in OWNERSHIP_LINES):
        return GROUPS[7]
    if cid.startswith("nationalised_"):
        return GROUPS[6]
    return CATEGORY_GROUP.get(category, GROUPS[5])


BILL_OK = have(F["bill by case (market)"], F["bill by case summary"])
lines, totals = {}, {}     # case -> Series(component -> GBP ex VAT); case -> (ex VAT from its own summary)
if BILL_OK:
    market = pd.read_csv(F["bill by case (market)"])
    by_case = read_json(F["bill by case summary"])["cases"]
    category = market.drop_duplicates("component_id").set_index("component_id")["category"]
    for c, _, src in CASES:
        if src == "market" and c in by_case:
            lines[c] = market[market.case_id.eq(c)].groupby("component_id")["annual_gbp_per_3100kwh_ex_vat"].sum()
            totals[c] = float(by_case[c]["ex_vat_gbp"])
    if have(F["P2 state-bid bills"], F["P2 state-bid summary"]):
        bids = pd.read_csv(F["P2 state-bid bills"])
        bid_summary = read_json(F["P2 state-bid summary"])["cases"]
        for s, c in BIDS.items():
            if c in bid_summary:
                lines[c] = bids[bids.case_id.eq(c)].groupby("component_id")["case_on_group"].sum()
                totals[c] = float(bid_summary[c]["total_gbp_inc_vat"]) / (1 + VAT)
    if have(F["P3 tariff lines (I1)"]):
        nat = pd.read_csv(F["P3 tariff lines (I1)"])
        category = pd.concat([category, nat.drop_duplicates("component_id").set_index("component_id")["category"]])
        category = category[~category.index.duplicated()]
        lines[P3_I1] = nat[nat.case_id.eq(P3_I1)].groupby("component_id")["annual_gbp_per_3100kwh_ex_vat"].sum()
        if have(F["P3 institution (I3)"], F["P3 institution summary"]):
            inst = pd.read_csv(F["P3 institution (I3)"]).set_index("component_id")
            inst_summary = read_json(F["P3 institution summary"])
            i3 = lines[P3_I1].copy()
            for comp, row in inst.iterrows():       # the measured lines and the institution line move; the rest is capacity-driven
                i3.loc[comp] = float(row.i3_gbp_per_3100kwh_ex_vat)
            lines[P3_I3] = i3
            totals[P3_I1], totals[P3_I3] = float(inst_summary["i1_ex_vat_gbp"]), float(inst_summary["i3_ex_vat_gbp"])
            for comp in inst.index:
                if comp not in category.index:
                    category.loc[comp] = "Cost of generation and storage (nationalised)"
    CASE_IDS = [c for c, _, _ in CASES if c in lines]
    stack = pd.DataFrame({c: lines[c].groupby(lambda cid: group_of(cid, category.get(cid, ""))).sum() for c in CASE_IDS}).reindex(GROUPS).fillna(0.0)
    check = pd.DataFrame({"lines sum, ex VAT": stack.sum(), "own summary, ex VAT": pd.Series(totals)})
    check["difference"] = check.iloc[:, 0] - check.iloc[:, 1]
    bad = check[check["difference"].abs() > 0.01]
    print("Identity: every case's lines sum to its own summary's total (GBP 0.01)." if bad.empty
          else f"WARNING: lines do not sum to the summary for {list(bad.index)}")
    INC = stack.sum() * (1 + VAT)
else:
    print("Bill outputs missing:", MISSING)
""")

code(r"""
if BILL_OK:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    xs = np.arange(len(CASE_IDS))
    for i, c in enumerate(CASE_IDS):
        up = down = 0.0
        for g in GROUPS:
            v = float(stack.loc[g, c])
            if abs(v) < 0.005:
                continue
            base = up if v > 0 else down
            ax.bar(i, v, bottom=base, width=0.62, color=GROUP_COLOUR[g], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if abs(v) > 40:
                ax.text(i, base + v / 2, f"{v:,.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if v > 0:
                up += v
            else:
                down += v
        ax.text(i, up + 12, gbp(INC[c]), ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
        if c != "private" and "private" in INC:
            ax.text(i, up + 50, f"{INC[c] - INC['private']:+,.0f}", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.axhline(0, color=INK_3, linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[c] for c in CASE_IDS], fontsize=9); ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(fmt_gbp); ax.grid(axis="x", visible=False)
    ax.set_ylim(min(stack.clip(upper=0).sum().min() * 1.4, -20), stack.clip(lower=0).sum().max() * 1.16)
    handles = [mpl.patches.Patch(color=GROUP_COLOUR[g], label=g) for g in GROUPS if stack.loc[g].abs().max() > 0.005]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8.5)
    title(ax, "The 2050 household bill by case", "GBP per year, 3,100 kWh household, 2025 real; bands ex VAT, totals inc VAT; change vs private above")
    caption(fig, "P2 bars at a share carry the state's balancing bids at cost on that share's own BM months (decision 46). P3 replaces the market lines "
                 "with the cost of generation and storage; I3 is the same fleet under central dispatch, with the FTI forecast-error charge at the 2050 fleet ratio (decision 18).", y=-0.04)
    plt.tight_layout(); plt.show()
""")

code(r"""
if BILL_OK:
    # Decision 12/12a: the heat-pump plus EV household is the 2050 headline, beside the FES per-unit build-up.
    # The projector writes each market case's archetypes; for the other cases the change is scaled by consumption,
    # which is how the projector's archetypes move (one standing charge, the change on the unit rate). Checked here.
    arche = {c: {a["archetype"]: a for a in by_case[c].get("household_archetypes", [])} for c in by_case}
    KEYS = [("heat_pump_ev_2050", "Heat pump + EV (G6 build-up)"), ("heat_pump_ev_2050_fes_per_unit", "Heat pump + EV (FES per unit)")]
    table = pd.DataFrame(index=[flat(c) for c in CASE_IDS])
    table["Bill ex VAT"] = [stack[c].sum() for c in CASE_IDS]
    table["Bill inc VAT"] = [INC[c] for c in CASE_IDS]
    table["Change vs private"] = [INC[c] - INC["private"] for c in CASE_IDS]
    worst = 0.0
    for key, name in KEYS:
        base = arche.get("private", {}).get(key)
        if not base:
            continue
        kwh = float(base["annual_consumption_kwh"])
        vals = []
        for c in CASE_IDS:
            derived = float(base["total_gbp_inc_vat"]) + (INC[c] - INC["private"]) * kwh / 3100.0
            own = arche.get(c, {}).get(key)
            if own:
                worst = max(worst, abs(derived - float(own["total_gbp_inc_vat"])))
            vals.append(float(own["total_gbp_inc_vat"]) if own else derived)
        table[f"{name}, {kwh:,.0f} kWh"] = vals
    display(table.style.format("£{:,.2f}").format("{:+,.2f}", subset=["Change vs private"]))
    print(f"Scaling check on the market cases: largest difference from the projector's own archetype GBP {worst:.2f}.")
""")

# =====================================================================================================
md(r"""
## 2. Who gains: the three-party columns

The Exchequer ledger's identity, against private and in system GBP a year: **consumers' change + the Exchequer's change
+ private investors' change = the resource saving**. Consumers are the bill's system-level change; investors are
private owners' rent at their required return; the resource saving is the real cost the economy avoids (mostly
the cheaper capital, net of corporation tax and the state's overhead). A column below zero is a loss to that party.

Since decisions 58 to 67 every private asset pays its way (section 2b), so the investor column is real rent, not
plant the private case left unpaid and the state appears to rescue.
""")

code(r"""
LEDGER_OK = have(F["three-party ledger"])
PARTIES = [("consumer_change_vs_private_gbp", "Consumers"), ("exchequer_net_vs_private_gbp", "Exchequer"),
           ("investor_change_vs_private_gbp", "Private investors"), ("resource_saving_vs_private_gbp", "Resource saving")]
if LEDGER_OK:
    ledger = pd.read_csv(F["three-party ledger"]).set_index("case_id")
    shown = [c for c, _, _ in CASES if c in ledger.index and c != "private"]
    parties = ledger.loc[shown, [p for p, _ in PARTIES]] / 1e9
    resid = ledger.loc[shown, "three_party_identity_residual_gbp"].abs().max()
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    xs, w = np.arange(len(shown)), 0.2
    colours = [SERIES[0], SERIES[1], SERIES[2], INK_2]
    for j, ((col, name), colour) in enumerate(zip(PARTIES, colours)):
        vals = parties[col].values
        ax.bar(xs + (j - 1.5) * w, vals, width=w * 0.92, color=colour, edgecolor=SURFACE, zorder=3, label=name)
        for x, v in zip(xs + (j - 1.5) * w, vals):
            ax.text(x, v + (0.15 if v >= 0 else -0.15), f"{v:,.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK_2)
    ax.axhline(0, color=INK_3, linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[c] for c in shown], fontsize=9); ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(fmt_bn); ax.grid(axis="x", visible=False); ax.legend(loc="upper left", ncol=4)
    title(ax, "Who gains in 2050: consumers, the Exchequer and private investors", "GBP bn a year vs private, 2024 real; the first three sum to the fourth")
    caption(fig, f"Identity residual, largest across these cases: GBP {resid:,.2f}. Investors on the required-return basis (the ledger's headline).")
    plt.tight_layout(); plt.show()
    per_hh = ledger.loc[shown, [f"{p}_per_household" for p, _ in PARTIES if f"{p}_per_household" in ledger.columns]]
    out = parties.rename(columns=dict(PARTIES)).round(2)
    out.index = [flat(c) for c in shown]
    display(out)
""")

# =====================================================================================================
md(r"""
### 2a. The risk-adjusted (Lucas) view

Decision 69. The headline counts the state's cheaper capital (`cost_of_capital_saving_gbp`) as a saving to the country.
The risk-adjusted view reads the same money as the market price of the risk taxpayers take on when the state finances
the plant: the resource saving loses it (`resource_saving_risk_adjusted_gbp`) and the Exchequer carries it
(`taxpayer_risk_bearing_gbp`, giving `exchequer_change_risk_adjusted_gbp`). Consumers and investors do not move, and
the identity still closes. **Neither reading is the answer**: the gap between the two resource-saving bars is the part
of the result that rests on how much of the private premium is risk society bears whoever owns the plant.

Below it, decision 68's co-investment guarantee: the co-owners' rate reduction in the halo sensitivity, carried by the
Exchequer outside the resource saving (zero in the central case, where each owner finances its own part).
""")

code(r"""
if LEDGER_OK and "resource_saving_risk_adjusted_gbp" in ledger.columns:
    risk = ledger.loc[shown].dropna(subset=["resource_saving_risk_adjusted_gbp"])
    rshown = list(risk.index)
    closes = (risk["consumer_change_vs_private_gbp"] + risk["exchequer_change_risk_adjusted_gbp"]
              + risk["investor_change_vs_private_gbp"] - risk["resource_saving_risk_adjusted_gbp"]).abs().max()
    fig, ax = plt.subplots(figsize=(13.5, 5))
    xs, w = np.arange(len(rshown)), 0.27
    pairs = [("resource_saving_vs_private_gbp", "Resource saving, headline", INK_2),
             ("resource_saving_risk_adjusted_gbp", "Resource saving, risk-adjusted", SERIES[6]),
             ("consumer_change_vs_private_gbp", "Consumers (either reading)", SERIES[0])]
    for j, (col, name, colour) in enumerate(pairs):
        vals = risk[col].astype(float).values / 1e9
        ax.bar(xs + (j - 1) * w, vals, width=w * 0.92, color=colour, edgecolor=SURFACE, zorder=3, label=name)
        for x, v in zip(xs + (j - 1) * w, vals):
            ax.text(x, v + (0.15 if v >= 0 else -0.15), f"{v:,.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK_2)
    ax.axhline(0, color=INK_3, linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[c] for c in rshown], fontsize=9); ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(fmt_bn); ax.grid(axis="x", visible=False); ax.legend(loc="upper left", ncol=3)
    title(ax, "The saving on two readings of the state's cheaper capital", "GBP bn a year vs private, 2024 real")
    caption(fig, f"Risk-adjusted: the cost-of-capital saving moved from the resource saving to the Exchequer as the price of the risk taxpayers bear "
                 f"(decision 69). Risk-adjusted identity residual, largest across these cases: GBP {closes:,.2f}.")
    plt.tight_layout(); plt.show()
    RISK_COLS = [("resource_saving_vs_private_gbp", "Resource saving, headline"),
                 ("cost_of_capital_saving_gbp", "of which cost-of-capital saving"),
                 ("resource_saving_risk_adjusted_gbp", "Resource saving, risk-adjusted"),
                 ("exchequer_net_vs_private_gbp", "Exchequer, headline"),
                 ("taxpayer_risk_bearing_gbp", "Taxpayer risk-bearing"),
                 ("exchequer_change_risk_adjusted_gbp", "Exchequer, risk-adjusted")]
    rt = risk[[c for c, _ in RISK_COLS if c in risk.columns]].astype(float) / 1e9
    rt = rt.rename(columns=dict(RISK_COLS)); rt.index = [flat(c) for c in rshown]
    display(rt.round(2))
elif LEDGER_OK:
    print("The ledger carries no risk-adjusted columns: public_ownership.risk_adjusted_view.enabled is off for this run "
          "(decision 69 turns it on for HT50 with the round-3 config).")

if LEDGER_OK:
    if "co_investment_implicit_guarantee_gbp" in ledger.columns:
        g = ledger.loc[shown, "co_investment_implicit_guarantee_gbp"].astype(float).fillna(0.0)
        print(f"Co-investors' implicit guarantee (decision 68), central run: largest GBP {g.abs().max() / 1e6:,.1f} m "
              "(zero under separate financing with no reduction; see the halo bound in section 3b).")
    else:
        print("No co_investment_implicit_guarantee_gbp column in this ledger (written before decision 68).")
""")

# =====================================================================================================
md(r"""
### 2b. Support for the private counterfactual

Decisions 58, 63, 64, 65 and 67. The private counterfactual must pay its way (decision 41): nobody builds or keeps a
plant that does not earn its required return. So every asset the hourly 2050 market leaves short is supported, and
each payment goes to the asset's owner, the same amount per asset in every case:

- **ancillary services**, a transfer inside the BSUoS lines the bill already pays (decision 65 rebases those lines so
  the services they hold equal the services paid);
- **EfW gate fees**, paid by waste producers, not the bill;
- **battery sub-hourly revenue**, +20% of wholesale plus balancing revenue for the intra-hour spreads hourly dispatch
  misses, paid by other market participants;
- **the battery top-up** (bill line `battery_revenue_top_up`), which brings each battery row to break even, because
  FES's battery fleet sits above the merchant equilibrium;
- **the life-extension operating top-up** (bill line `life_extension_operating_top_up`) for expired offshore wind,
  unabated biomass and expired BECCS, operating cost only.

The first table is the register's money (system GBP bn a year, the private case; the same assets in every market case).
The second is the bill lines by case. The third is the bill sensitivities on the private bill.
""")

code(r"""
SUPPORT_COLS = [("ancillary_services_revenue_gbp", "Ancillary services (paid out of BSUoS)"),
                ("ancillary_voltage_stability_revenue_gbp", "  of which voltage and stability"),
                ("efw_gate_fee_revenue_gbp", "EfW gate fees (paid by waste producers)"),
                ("battery_subhourly_revenue_gbp", "Battery sub-hourly revenue (paid by the market)"),
                ("battery_revenue_top_up_gbp", "Battery top-up (bill line)"),
                ("life_extension_top_up_gbp", "Life-extension operating top-up (bill line)"),
                ("battery_revenue_top_up_subhourly_high_gbp", "memo: battery top-up at the +40% sub-hourly case")]
if have(F["public account summary"]):
    pa = read_json(F["public account summary"]).get("cases", {})
    market_cases = [c for c, _, src in CASES if src == "market" and c in pa]
    present = [(col, name) for col, name in SUPPORT_COLS if any(col in pa[c] for c in market_cases)]
    if present:
        sup = pd.DataFrame({flat(c): {name: float(pa[c].get(col, np.nan)) / 1e9 for col, name in present} for c in market_cases})
        display(sup.round(3))
        spread = (sup.max(axis=1) - sup.min(axis=1)).abs().max()
        print(f"Largest difference between market cases: GBP {spread * 1e3:,.1f} m (the payments are per asset, so they should agree).")
    else:
        print("The public account carries no support-line columns: written before decision 58.")

SUPPORT_LINES = [("battery_revenue_top_up", "Battery top-up"), ("life_extension_operating_top_up", "Life-extension top-up"),
                 ("bsuos_other_services", "BSUoS non-constraint services"),
                 ("bsuos_voltage_and_stability_constraints", "BSUoS voltage and stability")]
if BILL_OK:
    bl = pd.DataFrame({flat(c): {name: lines[c].get(cid, np.nan) for cid, name in SUPPORT_LINES} for c in CASE_IDS})
    if bl.notna().any().any():
        print("Bill lines, GBP per 3,100 kWh household ex VAT (blank: the line is replaced, as under the P3 tariff):")
        display(bl.style.format("{:,.2f}", na_rep="-"))
    if not any(cid in lines.get("private", {}) for cid, _ in SUPPORT_LINES[:2]):
        print("The private bill carries no top-up line yet (bill written before decisions 58 and 63).")

if have(F["private bill summary"]):
    sens = (read_json(F["private bill summary"]).get("diagnostics") or {}).get("levy_and_cm_sensitivities_gbp_per_3100kwh_ex_vat") or {}
    if sens:
        s = pd.DataFrame({"GBP ex VAT": pd.Series(sens, dtype=float)})
        s["GBP inc VAT"] = s["GBP ex VAT"] * (1 + VAT)
        print("Bill sensitivities on the private bill (change, per 3,100 kWh; decision 60's electrolyser exemption, "
              "decision 65's ancillary low/high, decision 67's sub-hourly high where present):")
        display(s.style.format("{:+,.2f}"))
""")

# =====================================================================================================
md(r"""
## 3. Sensitivities

**3a. Named cases and the electrolyser exemption.** The public rate one point higher (decision 49 (h)) tests how far
the saving is the rate spread. The NESO wind ramp (decision 54) lowers private wind rates after 2035: each ramp case
is struck on the ramp's own calibrated strikes and compared with `private@private_rate_neso_wind_ramp` (the ledger's
`*_vs_comparator_*` columns), never with flat private. The electrolyser exemption (decision 60) takes electrolysers out
of the policy-levy bases: it moves private and every market case alike, and moves a tariff case's gap only through the
lines the tariff replaces.

**3b. The register bounds.** Each re-accounts the central months with one group-level block changed, and no re-solve:
a repowering share of 0.5 (decision 62), the co-owner halo and the pooled leakage test (decision 68), the CCC learning
rate (decision 11) and the strict economic life (Y10). Each is read against its own private case, then against the
central ledger's same case.

**3c. HPC and Sizewell C under P3** (decision 55): HPC paid in the carried 2030 annuity for the capacity the purchase
bought, Sizewell C acquired at its regulated asset value.
""")

code(r"""
if LEDGER_OK:
    named = [c for c in ledger.index.unique() if "@public_rate_" in c or "@private_rate_neso_wind_ramp" in c]
    if named:
        cols = [("bill_inc_vat", "Bill inc VAT"), ("bill_delta_vs_private_inc_vat", "vs private"),
                ("bill_delta_vs_comparator_gbp_inc_vat", "vs its comparator"),
                ("resource_saving_vs_private_gbp", "Resource saving vs private, bn"),
                ("resource_saving_vs_comparator_gbp", "Resource saving vs comparator, bn"),
                ("consumer_change_vs_comparator_gbp", "Consumers vs comparator, bn"),
                ("investor_change_vs_comparator_gbp", "Investors vs comparator, bn")]
        rows = {}
        for c in named + sorted({c.split("@")[0] for c in named if c.split("@")[0] in ledger.index}):
            r = ledger.loc[c]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            rows[f"{c}" + (f"  [comparator {r['comparator_case_id']}]" if "comparator_case_id" in r and isinstance(r["comparator_case_id"], str) else "")] = {
                name: (float(r[col]) / (1e9 if col.endswith("_gbp") else 1.0)) if col in r and pd.notna(r[col]) else np.nan
                for col, name in cols}
        display(pd.DataFrame(rows).T.style.format("{:,.2f}", na_rep="-"))
    else:
        print("No public-rate or wind-ramp cases in this ledger.")

if have(F["ledger summary"]):
    unc = read_json(F["ledger summary"]).get("uncertainties_by_case") or {}
    ex = {flat(c) if c in LABEL else c: v["levy_exemption_electrolysers_2050"] for c, v in unc.items()
          if isinstance(v, dict) and "levy_exemption_electrolysers_2050" in v and (c in LABEL)}
    if ex:
        t = pd.DataFrame({k: {"Private bill change": v["private_bill_change_gbp_inc_vat"],
                              "Case bill change": v["case_bill_change_gbp_inc_vat"],
                              "Gap change": v["gap_change_gbp_inc_vat"]} for k, v in ex.items()}).T
        print("Electrolysers exempt from the supplier levies (decision 60), GBP inc VAT per 3,100 kWh household:")
        display(t.style.format("{:+,.2f}"))
    else:
        print("The ledger summary carries no electrolyser-exemption entry (levy_exemption_electrolysers_2050).")
""")

code(r"""
# 3b. The register bounds, each against its own private case, then against the central ledger's same case.
BOUND_COLS = [("bill_inc_vat", "Bill inc VAT"), ("bill_delta_vs_private_inc_vat", "Bill vs own private"),
              ("consumer_change_vs_private_gbp", "Consumers, bn"), ("exchequer_net_vs_private_gbp", "Exchequer, bn"),
              ("investor_change_vs_private_gbp", "Investors, bn"), ("resource_saving_vs_private_gbp", "Resource saving, bn"),
              ("resource_saving_risk_adjusted_gbp", "Resource saving risk-adjusted, bn"),
              ("co_investment_implicit_guarantee_gbp", "Co-investment guarantee, bn")]
frames, deltas = {}, {}
for name, path in BOUND_LEDGERS.items():
    if not path.exists():
        continue
    b = pd.read_csv(path).drop_duplicates("case_id").set_index("case_id")
    keep = [c for c in b.index if c == "private" or c.split("@")[0] in (P1, P2) or c in LABEL]
    cols = [(c, n) for c, n in BOUND_COLS if c in b.columns]
    t = b.loc[keep, [c for c, _ in cols]].astype(float)
    t[[c for c, _ in cols if c.endswith("_gbp")]] /= 1e9
    frames[name] = t.rename(columns=dict(cols))
    if LEDGER_OK:
        common = [c for c in keep if c in ledger.index]
        central = ledger.loc[common, [c for c, _ in cols]].astype(float)
        central[[c for c, _ in cols if c.endswith("_gbp")]] /= 1e9
        deltas[name] = (t.loc[common] - central).rename(columns=dict(cols))
if frames:
    print("Register bounds as solved (each against its own private case; GBP inc VAT per 3,100 kWh, system GBP bn a year):")
    display(pd.concat(frames).style.format("{:,.2f}", na_rep="-"))
    if deltas:
        print("Bound minus the central ledger, same case:")
        d = pd.concat(deltas)
        display(d.style.format("{:+,.2f}", na_rep="-"))
        col = "Bill vs own private"
        if col in d.columns:
            dd = d[col].drop(index="private", level=1, errors="ignore").dropna()
            if len(dd):
                fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(dd))))
                ys = np.arange(len(dd))
                ax.barh(ys, dd.values, color=[SAVING if v < 0 else COST for v in dd.values], height=0.6, zorder=3)
                ax.set_yticks(ys); ax.set_yticklabels([f"{a}: {flat(c) if c in LABEL else c}" for a, c in dd.index], fontsize=8)
                ax.invert_yaxis(); ax.axvline(0, color=INK_3, linewidth=0.8); ax.grid(axis="y", visible=False)
                ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:+,.1f}"))
                title(ax, "How each bound moves a case's saving", "change in the bill gap to private vs the central run, GBP inc VAT per 3,100 kWh")
                plt.tight_layout(); plt.show()
missing = [n for n, p in BOUND_LEDGERS.items() if not p.exists()]
if missing:
    print("Register bounds not on disk:", *missing, sep="\n  ")
""")

code(r"""
# 3c. Decision 55 in the P3 tariff: what was carried from the 2030 purchase and what Sizewell C costs at its RAV.
if have(F["P3 tariff summary"]):
    ts = read_json(F["P3 tariff summary"])
    diag = (((ts.get("cases") or {}).get(P3_I1) or {}).get("diagnostics") or ts.get("diagnostics") or {})
    d55 = diag.get("decision_55")
    if d55:
        flat55 = {k: v for k, v in d55.items() if not isinstance(v, dict)}
        for k, v in d55.items():
            if isinstance(v, dict):
                flat55.update({f"{k}: {kk}": vv for kk, vv in v.items()})
        print("Decision 55 (P3 tariff diagnostics; GBP are 2024 real, MW as tagged in the register):")
        display(pd.Series(flat55, dtype=object).to_frame("value"))
    else:
        print("The P3 tariff carries no decision_55 diagnostics (written before decision 55).")
    sup = {k: v for k, v in diag.items() if k.startswith(("private_battery", "private_top_up", "private_ancillary", "private_efw"))}
    if sup:
        print("Support lines the tariff replaces or credits (private case, and the part on owned plant), GBP bn:")
        display((pd.Series(sup, dtype=float) / 1e9).round(3).to_frame("GBP bn"))
""")

# =====================================================================================================
md(r"""
## 4. 2030 against 2050

The same cases in both snapshot years. In 2030 most of the public saving is the legacy fleet's rent (the RO and early
CfDs at their peak), which only P3 captures, by buying the fleet. By 2050 that rent has largely run off while the
2030 purchase annuity is still paid, and most of the fleet was built after 2025: the saving becomes the cost of
financing the new build, which P2 captures as fully as P3. The right panel shows that change of source from the
pathway's own snapshot years (consumers plus the Exchequer, by mechanism).

The two years differ in basis as well as in time: CP30 against FES HT 2050, and the 2030 cost table against the full
DESNZ basis (decision 26). The finance rules differ too until the 2030 session lands its handovers (decision 48's
lives and decision 62's onshore rule; the ancillary and battery support lines, keyed in 2030 and off by default); the
merchant level annuity (decision 66) and the life-extension top-up are 2050 only. The 2030 ledger is read as it stands
on disk; its write time is in the provenance box.
""")

code(r"""
PAIRS = [("Private", "private", "private"), ("P1", P1, P1), ("P2 market bids", P2, P2),
         ("P2 state bids", "public_ownership@state_bids_at_cost", BIDS[100]), ("P3 I1", P3_I1, P3_I1), ("P3 I3", P3_I3, P3_I3)]
if have(F["2030 ledger"]) and LEDGER_OK:
    cols = ["case_id", "bill_inc_vat"] + [p for p, _ in PARTIES]
    led30 = pd.read_csv(F["2030 ledger"], usecols=lambda c: c in cols).drop_duplicates("case_id").set_index("case_id")
    pairs = [(n, a, b) for n, a, b in PAIRS if a in led30.index and b in ledger.index]
    yy = pd.DataFrame({"2030 bill": [led30.loc[a, "bill_inc_vat"] for _, a, _ in pairs],
                       "2050 bill": [ledger.loc[b, "bill_inc_vat"] for _, _, b in pairs],
                       "2030 resource saving, bn": [led30.loc[a, "resource_saving_vs_private_gbp"] / 1e9 for _, a, _ in pairs],
                       "2050 resource saving, bn": [ledger.loc[b, "resource_saving_vs_private_gbp"] / 1e9 for _, _, b in pairs],
                       "2030 consumers, bn": [led30.loc[a, "consumer_change_vs_private_gbp"] / 1e9 for _, a, _ in pairs],
                       "2050 consumers, bn": [ledger.loc[b, "consumer_change_vs_private_gbp"] / 1e9 for _, _, b in pairs]},
                      index=[n for n, _, _ in pairs])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    xs = np.arange(len(yy))
    ax = axes[0]
    for j, (col, colour) in enumerate([("2030 bill", NEUTRAL), ("2050 bill", SERIES[0])]):
        ax.bar(xs + (j - 0.5) * 0.38, yy[col], width=0.36, color=colour, edgecolor=SURFACE, zorder=3, label=col[:4])
        for x, v in zip(xs + (j - 0.5) * 0.38, yy[col]):
            ax.text(x, v + 8, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5, color=INK_2, rotation=90)
    ax.set_ylim(0, yy[["2030 bill", "2050 bill"]].max().max() * 1.35)
    ax.set_xticks(xs); ax.set_xticklabels(yy.index, fontsize=8.5, rotation=30, ha="right"); ax.yaxis.set_major_formatter(fmt_gbp)
    ax.grid(axis="x", visible=False); ax.legend(loc="upper right", ncol=2)
    title(ax, "The bill", "inc VAT, 3,100 kWh, 2025 real", tagged=False)
    ax = axes[1]
    sub = yy.drop(index="Private", errors="ignore")
    xs2 = np.arange(len(sub))
    for j, (col, colour, hatch) in enumerate([("2030 resource saving, bn", NEUTRAL, None), ("2050 resource saving, bn", SERIES[0], None),
                                               ("2030 consumers, bn", "none", "////"), ("2050 consumers, bn", "none", "xxxx")]):
        ax.bar(xs2 + (j - 1.5) * 0.2, sub[col], width=0.19, facecolor=colour, edgecolor=INK_3 if hatch else SURFACE, hatch=hatch, zorder=3,
               label=col.replace(", bn", ""))
    ax.axhline(0, color=INK_3, linewidth=0.8)
    ax.set_xticks(xs2); ax.set_xticklabels(sub.index, fontsize=8.5, rotation=30, ha="right"); ax.yaxis.set_major_formatter(fmt_bn)
    ax.grid(axis="x", visible=False); ax.legend(loc="upper left", fontsize=8)
    title(ax, "The saving", "GBP bn a year vs private, 2024 real", tagged=False)
    ax = axes[2]
    if have(F["pathway summary"]):
        psum = read_json(F["pathway summary"])["cases"]
        mech = ["financing", "legacy", "market_and_other"]
        mcol = {"financing": SERIES[0], "legacy": SERIES[1], "market_and_other": NEUTRAL}
        rows = []
        for cid in (BIDS[100], P3_I1):
            if cid in psum:
                ann = psum[cid]["annual"]
                for yr in ("2030", "2050"):
                    rows.append((f"{psum[cid]['label']}\n{yr}", {m: (ann["consumers"][yr][m] + ann["exchequer"][yr][m]) / 1e9 for m in mech}))
        for i, (name, vals) in enumerate(rows):
            up = down = 0.0
            for m in mech:
                v = vals[m]
                base = up if v >= 0 else down
                ax.bar(i, v, bottom=base, width=0.6, color=mcol[m], edgecolor=SURFACE, zorder=3, label=m.replace("_", " ") if i == 0 else None)
                up, down = (up + v, down) if v >= 0 else (up, down + v)
            ax.text(i, up + 0.2, f"{sum(vals.values()):,.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(0, color=INK_3, linewidth=0.8)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=8.5); ax.yaxis.set_major_formatter(fmt_bn)
        ax.grid(axis="x", visible=False); ax.legend(loc="lower left", fontsize=8)
        title(ax, "Where the gain comes from", "consumers + Exchequer, GBP bn a year, by mechanism", tagged=False)
    fig.suptitle("2030 against 2050" + TAG, x=0.01, ha="left", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.show()
    display(yy.style.format("{:,.2f}"))
""")

code(r"""
if have(F["2030 private bill lines"], F["private bill lines"]):
    c30 = pd.read_csv(F["2030 private bill lines"])
    c50 = pd.read_csv(F["private bill lines"])
    by_year = pd.DataFrame({
        "2030 (CP30)": c30.groupby(c30.apply(lambda r: group_of(r.component_id, r.category), axis=1))["annual_gbp_per_3100kwh_ex_vat"].sum(),
        "2050 (FES HT)": c50.groupby(c50.apply(lambda r: group_of(r.component_id, r.category), axis=1))["annual_gbp_per_3100kwh_ex_vat"].sum(),
    }).reindex(GROUPS).fillna(0.0)
    by_year = by_year[(by_year.abs() > 0.005).any(axis=1)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, col in enumerate(by_year.columns):
        up = 0.0
        for g, v in by_year[col].items():
            ax.bar(i, v, bottom=up if v > 0 else 0, width=0.5, color=GROUP_COLOUR[g], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=g if i == 0 else None)
            if v > 25:
                ax.text(i, up + v / 2, f"{v:,.0f}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
            up += max(v, 0)
        ax.text(i, up + 10, gbp(by_year[col].sum() * (1 + VAT)) + " inc VAT", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(by_year.columns); ax.set_xlim(-0.6, 2.6)
    ax.yaxis.set_major_formatter(fmt_gbp); ax.grid(axis="x", visible=False); ax.legend(loc="center right", fontsize=8.5)
    title(ax, "The private bill, 2030 and 2050", "GBP per year ex VAT by band, 3,100 kWh, 2025 real")
    plt.tight_layout(); plt.show()
""")

# =====================================================================================================
md(r"""
## 5. The pathway 2030-2050

Year by year between the two solved snapshots (plan Y15). Financing is exact from each vintage's own rate, share and
term; P3's legacy line is the acquired fleet's cash less the 2030 purchase annuity, which is paid flat to 2059; the
rest is interpolated between the snapshots, and each snapshot year reproduces its own ledger. Bars are the public
gain (consumers plus the Exchequer) by mechanism; the dark line is the resource saving.

**Beyond 2050** is a separate line, never inside the 2030-2050 totals: the financing saving still to come on plant
built by 2050, less P3's annuity from 2051 to 2059, held at 2050 levels and discounted to 2030.
""")

code(r"""
PATH_OK = have(F["pathway 2030-2050"], F["pathway summary"])
if PATH_OK:
    path = pd.read_csv(F["pathway 2030-2050"])
    psum_all = read_json(F["pathway summary"])
    psum = psum_all["cases"]
    core = path[~path.line.str.contains(":") & path.mechanism.ne("beyond_2050") & path.year.between(2030, 2050)]
    public = (core[core.party.isin(["consumers", "exchequer"])].groupby(["case_id", "year", "mechanism"])["gbp"].sum() / 1e9).unstack("mechanism").fillna(0.0)
    resource = core[core.party.eq("resource")].groupby(["case_id", "year"])["gbp"].sum() / 1e9
    order = [c for c in [P1, BIDS[25], BIDS[50], BIDS[75], BIDS[100], P3_I1, P3_I3] if c in psum]
    mech = [m for m in ["financing", "legacy", "market_and_other"] if m in public.columns]
    mcol = {"financing": SERIES[0], "legacy": SERIES[1], "market_and_other": NEUTRAL}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))   # limits shared by set_ylim, not sharey (the last panel is horizontal)
    lo = min(public.clip(upper=0).sum(axis=1).min(), resource.min()) * 1.15
    hi = max(public.clip(lower=0).sum(axis=1).max(), resource.max()) * 1.15
    for ax, cid in zip(axes.flat, order):
        p = public.loc[cid]
        up = np.zeros(len(p)); down = np.zeros(len(p))
        for m in mech:
            v = p[m].values
            ax.bar(p.index, v, bottom=np.where(v >= 0, up, down), width=0.8, color=mcol[m], edgecolor=SURFACE, linewidth=0.5, zorder=3,
                   label=m.replace("_", " "))
            up, down = up + np.clip(v, 0, None), down + np.clip(v, None, 0)
        ax.plot(resource.loc[cid].index, resource.loc[cid].values, color=INK, linewidth=2, zorder=4, label="resource saving")
        ax.axhline(0, color=INK_3, linewidth=0.8)
        t = psum[cid]["public_gain_consumers_plus_exchequer"]["total"]
        r = psum[cid]["parties"]["resource"]["total"]
        title(ax, psum[cid]["label"], f"public gain {t['undiscounted'] / 1e9:,.0f}bn, NPV {t['npv_stpr_3p5'] / 1e9:,.0f}bn; resource NPV {r['npv_stpr_3p5'] / 1e9:,.0f}bn",
              tagged=False)
        ax.set_ylim(lo, hi); ax.yaxis.set_major_formatter(fmt_bn); ax.grid(axis="x", visible=False)
    ax = axes.flat[len(order)]
    b_res = [psum[c]["beyond_2050"]["parties"]["resource"]["total"]["npv_stpr_3p5"] / 1e9 for c in order]
    b_pub = [(psum[c]["beyond_2050"]["parties"]["consumers"]["total"]["npv_stpr_3p5"]
              + psum[c]["beyond_2050"]["parties"]["exchequer"]["total"]["npv_stpr_3p5"]) / 1e9 for c in order]
    in_res = [psum[c]["parties"]["resource"]["total"]["npv_stpr_3p5"] / 1e9 for c in order]
    ys = np.arange(len(order))
    ax.barh(ys + 0.2, in_res, height=0.38, color=INK_2, label="2030-2050, resource NPV")
    ax.barh(ys - 0.2, b_res, height=0.38, color=SERIES[6], label="beyond 2050, resource NPV")
    ax.plot(b_pub, ys - 0.2, "o", color=SERIES[3], markersize=8, markeredgecolor=SURFACE, label="beyond 2050, consumers + Exchequer NPV")
    ax.set_yticks(ys); ax.set_yticklabels([psum[c]["label"] for c in order], fontsize=8.5); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}")); ax.set_xlabel("GBP bn")
    ax.grid(axis="y", visible=False); ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), fontsize=7.5)
    ax.tick_params(axis="y", labelleft=True)
    title(ax, "2030-2050 against beyond 2050", "NPV at 3.5% to 2030, GBP bn", tagged=False)
    axes.flat[0].legend(loc="upper left", fontsize=7.5)
    for ax in axes.flat[len(order) + 1:]:
        ax.set_visible(False)
    fig.suptitle("The annual public benefit 2030-2050 by mechanism" + TAG, x=0.01, ha="left", fontsize=12, fontweight="bold")
    caption(fig, "GBP bn a year vs private, 2024 real. P2 cases are on the state-bids basis; P2 at a share has no 2030 snapshot and takes share x P2 there "
                 "(decision 17). Beyond 2050 is never inside the 2030-2050 totals (decision 46 (b)).", y=0.0)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95)); plt.show()
""")

code(r"""
if PATH_OK:
    def bn(d, k="undiscounted"):
        return d[k] / 1e9
    rows = {}
    for cid in order:
        e = psum[cid]
        res, con = e["parties"]["resource"]["total"], e["parties"]["consumers"]["total"]
        hh = e["parties"]["consumers"]["per_household_3100kwh_gbp2025"]
        b = e["beyond_2050"]["parties"]
        rows[e["label"]] = {
            "Resource saving, bn": bn(res), "NPV 3.5%": bn(res, "npv_stpr_3p5"), "NPV 3.0%": bn(res, "npv_low_3p0"), "NPV 5.0%": bn(res, "npv_high_5p0"),
            "Consumers, bn": bn(con), "Consumers NPV 3.5%": bn(con, "npv_stpr_3p5"),
            "Per household, GBP": hh["undiscounted"], "Per household NPV, GBP": hh["npv_stpr_3p5"],
            "Beyond 2050 resource NPV, bn": bn(b["resource"]["total"], "npv_stpr_3p5"),
            "Beyond 2050 consumers + Exchequer NPV, bn": bn(b["consumers"]["total"], "npv_stpr_3p5") + bn(b["exchequer"]["total"], "npv_stpr_3p5"),
        }
    totals_table = pd.DataFrame(rows).T
    print("2030-2050 totals (anchored; the headline, decision 46 (d)). Undiscounted and NPV to 2030; per household per 3,100 kWh, GBP 2025.")
    display(totals_table.style.format("{:,.1f}").format("{:,.0f}", subset=["Per household, GBP", "Per household NPV, GBP"]))
    anch = psum_all.get("anchoring", {})
    print(f"Anchoring: {anch.get('rule', '')}; all pass: {anch.get('all_pass')}; worst difference GBP {anch.get('worst_difference_gbp', float('nan')):,.0f}.")
""")

# =====================================================================================================
md(r"""
## 6. The balancing mechanism by month: market against central dispatch

Load shed in the BM and the BM's constraint cost (accepted offers less bids at their prices, excluding the shedding),
month by month, for the market institution (Stage 1 copperplate day-ahead plus the BM) and the central-dispatch year
(one network-constrained schedule plus a real-time correction). The markers are the market bill's own constraint
line by month (it also carries interconnector re-trading and CfD curtailment compensation). The winter shedding that
remains is behind the southern and Welsh boundaries, held at their 2045 capability (decision 36).
""")

code(r"""
def bm_month(scenario):
    path = MK / f"{scenario}_constraint_costs.csv"
    if not path.exists():
        return None
    c = pd.read_csv(path)
    c = c[c.carrier.ne("TOTAL")]
    shed = c.loc[c.carrier.eq("load_shedding"), "increase_MWh"].sum() / 1e3
    return shed, c.loc[c.carrier.ne("load_shedding"), "net_cost"].sum() / 1e6


series = {"market": f"{STUDY}_{{m}}_market", "central": f"{STUDY}_{{m}}_central_dispatch"}
series |= {f"P2 {s}% state bids": f"{STUDY}_{{m}}_p2_bids_{s}pct" for s in SHARES}
bm = {}
for name, pattern in series.items():
    rows = {m: bm_month(pattern.format(m=m)) for m in MONTHS}
    rows = {m: v for m, v in rows.items() if v is not None}
    if rows:
        bm[name] = pd.DataFrame(rows, index=["shed_gwh", "cost_m"]).T.reindex(MONTHS)
if {"market", "central"} <= set(bm):
    bill_basis = None
    if have(F["market BM costs (bill basis)"]):
        bb = pd.read_csv(F["market BM costs (bill basis)"])
        bb["month"] = bb.scenario.str.replace(f"{STUDY}_", "", regex=False)
        bill_basis = bb.set_index("month")["constraint_cost_gbp"].reindex(MONTHS) / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    xs = np.arange(12)
    for ax, key, unit, fmt in [(axes[0], "shed_gwh", "GWh", "{x:,.0f}"), (axes[1], "cost_m", "GBP m", "£{x:,.0f}m")]:
        for j, (name, colour) in enumerate([("market", SERIES[0]), ("central", SERIES[1])]):
            ax.bar(xs + (j - 0.5) * 0.38, bm[name][key], width=0.36, color=colour, edgecolor=SURFACE, zorder=3,
                   label=f"{name}: year {bm[name][key].sum():,.0f} {unit}")
        if key == "cost_m":
            for s, ls in zip(SHARES, [":", "-.", "--", "-"]):
                name = f"P2 {s}% state bids"
                if name in bm:
                    ax.plot(xs, bm[name][key], color=SERIES[6], linewidth=1.2, linestyle=ls, zorder=4,
                            label=f"{name}: year {bm[name][key].sum():,.0f} {unit}")
            if bill_basis is not None:
                ax.plot(xs, bill_basis.values, "D", color=INK, markersize=6, zorder=5,
                        label=f"market bill's constraint line: year {bill_basis.sum():,.0f} {unit}")
        ax.axhline(0, color=INK_3, linewidth=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(MONTHS); ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(fmt))
        ax.grid(axis="x", visible=False); ax.legend(loc="upper center", fontsize=8)
    title(axes[0], "BM load shed", "GWh per month")
    title(axes[1], "BM constraint cost, excluding shedding", "GBP m per month, 2024 real")
    plt.tight_layout(); plt.show()
    table = pd.concat({k: v for k, v in bm.items()}, axis=1)
    table.loc["Year"] = table.sum()
    display(table.round(1))
else:
    print("Monthly BM tables not all on disk:", sorted(set(series) - set(bm)))
""")

# =====================================================================================================
md(r"""
## 7. European prices, and validation against FES ES1

**European prices (decision 29).** The central foreign prices are TYNDP 2024 Distributed Energy 2050 with IC1's
tranches. Global Ambition and the TYNDP 2026 draft National Trends+ re-price the same network at solve time, first on
the gate week (10-16 January). A gate week is not a year: its continental discount differs from the annual one, so read
the week for direction and the NT+ year (where solved) for size.
""")

code(r"""
GATES = {"TYNDP 2024 DE (base)": "HT50_1week_gate_d40", "TYNDP 2024 Global Ambition": "HT50_1week_gate_d40_ga",
         "TYNDP 2026 NT+": "HT50_1week_gate_d40_tyndp26"}


def stage1(scenario):
    out = {}
    price = MK / f"{scenario}_wholesale_price.csv"
    if price.exists():
        p = pd.read_csv(price)
        out["Stage 1 GB price, mean GBP/MWh"] = p["wholesale_price"].mean()
        if "hydrogen_price" in p:
            out["Hydrogen value, mean GBP/MWh_H2"] = p["hydrogen_price"].mean()
    links = MK / f"{scenario}_wholesale_links.csv"
    if links.exists():   # IC_ links: positive is an import into GB
        ic = pd.read_csv(links, index_col=0, usecols=lambda c: c == "snapshot" or str(c).startswith("IC_"))
        out["Stage 1 imports, GWh"] = ic.clip(lower=0).sum().sum() / 1e3
        out["Stage 1 exports, GWh"] = -ic.clip(upper=0).sum().sum() / 1e3
    b = bm_month(scenario)
    if b:
        out["BM shed, GWh"], out["BM constraint cost excl. shedding, GBP m"] = b
    return out


gate = pd.DataFrame({k: stage1(v) for k, v in GATES.items()})
if not gate.empty:
    display(gate.style.format("{:,.1f}"))
    metrics = [m for m in ["Stage 1 GB price, mean GBP/MWh", "Stage 1 imports, GWh", "Stage 1 exports, GWh",
                           "BM constraint cost excl. shedding, GBP m"] if m in gate.index]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.8))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        vals = gate.loc[m]
        ax.bar(range(len(vals)), vals.values, color=[SERIES[0], SERIES[2], SERIES[3]][:len(vals)], width=0.6, edgecolor=SURFACE, zorder=3)
        for i, v in enumerate(vals.values):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels([k.replace("TYNDP ", "").replace(" (base)", "\n(base)") for k in vals.index], fontsize=8)
        ax.grid(axis="x", visible=False); title(ax, m.split(",")[0], m.split(",")[1].strip() if "," in m else None, tagged=False)
    fig.suptitle("Gate week 10-16 January: European price sensitivities" + TAG, x=0.01, ha="left", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.9)); plt.show()
else:
    print("No gate-week outputs on disk.")
""")

code(r"""
# The NT+ year (decision 29 follow-up): month by month against the base (TYNDP DE) market year, where solved.
nt = {m: stage1(f"{STUDY}_{m}_tyndp2026_prices") for m in MONTHS}
nt = {m: v for m, v in nt.items() if "BM constraint cost excl. shedding, GBP m" in v}
if nt:
    de = {m: stage1(f"{STUDY}_{m}_market") for m in nt}
    cmp = pd.concat({"DE (base)": pd.DataFrame(de).T, "NT+": pd.DataFrame(nt).T}, axis=1).reindex([m for m in MONTHS if m in nt])
    print(f"NT+ months solved through the BM: {len(nt)}/12" + ("" if len(nt) == 12 else " (a partial year: read month by month, not as an annual total)"))
    keep = ["Stage 1 GB price, mean GBP/MWh", "Stage 1 imports, GWh", "Stage 1 exports, GWh", "BM constraint cost excl. shedding, GBP m", "BM shed, GWh"]
    cmp = cmp.loc[:, [(s, k) for k in keep for s in ("DE (base)", "NT+") if (s, k) in cmp.columns]]
    if len(nt) == 12:
        tot = cmp.sum(); tot[[c for c in cmp.columns if "price" in c[1]]] = cmp[[c for c in cmp.columns if "price" in c[1]]].mean()
        cmp.loc["Year"] = tot
    display(cmp.round(1))
else:
    print("No NT+ year months solved through the BM yet.")
""")

md(r"""
**Validation (Y8 acceptance items 2 and 3).** The final (BM) dispatch against FES 2025 ES1 for calendar 2050. Each WARN
carries an explanation built from the run's own numbers; a WARN without one fails the validation rule. The IC1
diagnostics cover a winter and a summer month; storage churn is simultaneous charge and discharge in the BM.
""")

code(r"""
if have(F["ES1 dispatch validation"], F["ES1 validation summary"]):
    val = pd.read_csv(F["ES1 dispatch validation"])
    vs = read_json(F["ES1 validation summary"])
    print(f"Status: {vs['status_counts']}; unexplained WARNs: {vs['unexplained_warns']} ({vs['months']} months, {vs['hours']:,.0f} h)")
    disp = val[val.component.eq("dispatched_energy") & val.target_gwh.notna() & val.model_gwh.notna()].copy()
    disp = disp.sort_values("target_gwh")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(disp))))
    ys = np.arange(len(disp))
    ax.barh(ys + 0.2, disp.target_gwh / 1e3, height=0.38, color=NEUTRAL, label="FES ES1")
    ax.barh(ys - 0.2, disp.model_gwh / 1e3, height=0.38, color=SERIES[0], label="model, final (BM)")
    if "stage1_gwh" in disp:
        ax.plot(disp.stage1_gwh / 1e3, ys - 0.2, "|", color=INK, markersize=9, label="model, Stage 1")
    for y, (_, r) in zip(ys, disp.iterrows()):
        ax.text(max(r.target_gwh, r.model_gwh) / 1e3 + 3, y, f"{r.status}  {r.relative_difference:+.0%}" if pd.notna(r.relative_difference) else r.status,
                va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(ys); ax.set_yticklabels(disp["item"], fontsize=8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("TWh, calendar 2050"); ax.legend(loc="lower right")
    title(ax, "Dispatch against FES 2025 ES1", "dispatched energy by line; status and relative difference at right")
    plt.tight_layout(); plt.show()
    warns = val[val.status.eq("WARN")][["component", "item", "target_gwh", "stage1_gwh", "model_gwh", "relative_difference", "explanation"]]
    with pd.option_context("display.max_colwidth", 160):
        display(warns.round(1))
    cur = pd.DataFrame(vs.get("curtailment_split_gwh", {})).T
    if not cur.empty:
        display((cur / 1e3).round(1).rename(columns=lambda c: c.replace("_mwh", ", TWh")))
""")

code(r"""
if have(F["IC1 diagnostics"]):
    ic = read_json(F["IC1 diagnostics"])
    display(pd.DataFrame(ic["totals"]).T.rename(columns=lambda c: c.replace("_gwh", ", GWh").replace("_gbp", ", GBP")).round(0))
if have(F["BM storage churn"]):
    ch = read_json(F["BM storage churn"])
    yr = pd.DataFrame(ch["year_by_carrier"]).set_index("carrier")
    yr = yr[yr.simultaneous_gwh > 0].sort_values("simultaneous_gwh", ascending=False)
    print(f"BM storage churn, the year: {yr.simultaneous_gwh.sum():,.1f} GWh simultaneous, {yr.energy_lost_gwh.sum():,.1f} GWh lost through the round trip.")
    display(yr[["simultaneous_gwh", "energy_lost_gwh", "simultaneous_share_of_charge"]].style.format(
        {"simultaneous_gwh": "{:,.1f}", "energy_lost_gwh": "{:,.1f}", "simultaneous_share_of_charge": "{:.3%}"}))
if MISSING:
    print("Files not found (their sections were skipped):", *MISSING, sep="\n  ")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT.relative_to(ROOT)} ({len(cells)} cells)")
