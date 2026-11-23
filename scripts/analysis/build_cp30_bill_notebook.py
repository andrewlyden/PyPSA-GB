"""Generate notebooks/cp30_2030_consumer_bill_figures.ipynb (bill, provenance, ownership, cost recovery, acquisition).

Run from anywhere:  python scripts/analysis/build_cp30_bill_notebook.py
then execute:       jupyter nbconvert --to notebook --execute --inplace notebooks/cp30_2030_consumer_bill_figures.ipynb
The notebook reads the best bill on disk and the finance outputs in resources/public_ownership/.
"""
import argparse
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "cp30_2030_consumer_bill_figures.ipynb"
# Network bounds (18-19 Oct 2026): the run group and forecast case the notebook reads are arguments
# (public_ownership.report_group and its forecast case), written into the notebook's setup cell.
_ARGS = argparse.ArgumentParser(description="Generate the CP30 2030 bill figures notebook")
_ARGS.add_argument("--group", default="cp30_2030_cp30_network")
_ARGS.add_argument("--case", default="CP30_2030_cp30_network")
_ARGS.add_argument("--pattern", default="CP30_FFR_2030_{month}_cp30_network",
                   help="the group's scenario_pattern ({month} = Jan..Dec): the months whose dispatch the notebook reads")
_ARGS, _ = _ARGS.parse_known_args()

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =====================================================================================================
md(r"""
# CP30 2030 consumer bill, and what public ownership does to it

Annual electricity bill of a 3,100 kWh household under CP30 Further Flex and Renewables 2030 (twelve monthly
ETYS market runs, 2013 weather). GBP per year, ex VAT unless stated, 2025 real prices.

1. **The bill**: at a glance, every component, and how it compares with the electricitybills.uk 2030 forecast.
2. **Public ownership**: three financing cases on the same run; what is routed to the bill today.
3. **Impact map**: every line, every case, and the largest plausible effect.
4. **Public cost recovery versus pay-as-bid**: the wholesale line under different settlement rules.
5. **Buying the legacy fleet**: what acquisition would unlock, as a function of the price paid (hook for the acquisition-cost work).

**Read the provenance box first.** The notebook uses the final twelve-month brownfield bill when it is present and
newer than any provisional file; otherwise it falls back to the provisional bill and tags every title as such.
""")

code(r"""
from __future__ import annotations

import json
import sys
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
# The kernel starts in notebooks/ (nbconvert, Jupyter), so the repository's own modules need the root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ANALYSIS = ROOT / "resources" / "analysis"
PUBLIC = ROOT / "resources" / "public_ownership"
BASELINE = ANALYSIS / "baselines" / "pre_brownfield_2026-09-10"
PROVISIONAL = ANALYSIS / "baselines" / "provisional_2026-09-11"
CASE = "__REPORT_CASE__"
GROUP = "__REPORT_GROUP__"
MONTH_PATTERN = "__REPORT_PATTERN__"          # the group's twelve months, {month} = Jan..Dec
VALUE = "annual_gbp_per_3100kwh_ex_vat"

# ---- style ----------------------------------------------------------------
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#84837c", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
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
# Groups are data (data/market/consumer_bill_benchmarks/bill_groupings.csv, 6 Oct 2026): the bill is shown in
# Ofgem's price-cap categories at the current cap release, the ownership impact map in what each cost pays for.
from scripts.consumer_bill.bill_groupings import annotate as annotate_groups, group_order, view_title
CATEGORY_ORDER = group_order("ofgem")
CATEGORY_TITLE = view_title("ofgem")
FUNCTIONAL_ORDER = group_order("functional")
CATEGORY_COLOUR = dict(zip(CATEGORY_ORDER, SERIES))
gbp = lambda v, d=0: f"£{v:,.{d}f}"
fmt_axis = mpl.ticker.StrMethodFormatter("£{x:,.0f}")


def title(ax, text, sub=None, tagged=True):
    # Bold left title with an optional grey subtitle a fixed 4 pt above the axes.
    ax.set_title(text + (TAG if tagged else ""), pad=20 if sub else 8, color=INK)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    fontsize=9, color=INK_3, va="bottom", ha="left", annotation_clip=False)


def caption(fig, text, y=-0.01):
    fig.text(0.01, y, text, fontsize=8.5, color=INK_3, va="top")


# ---- load the best bill on disk ------------------------------------------------------------------
# Three tiers: the final rebuilt twelve-month bill, the provisional new-fleet bill saved while the
# rebuild was solving (rebuilt months modelled, missing months scaled), and the pre-rebuild baseline.
# The final tier wins whenever its components and summary both exist and are newer than every
# provisional file; REBUILD_START is only used by the provenance box to count rebuilt networks.
REBUILD_START = datetime(2026, 9, 10, 19, 52)


def load_bill():
    live = (ANALYSIS / f"{CASE}_consumer_bill_future_components.csv", ANALYSIS / f"{CASE}_consumer_bill_future_summary.json")
    prov = (PROVISIONAL / f"{CASE}_consumer_bill_PROVISIONAL_components.csv", PROVISIONAL / f"{CASE}_consumer_bill_PROVISIONAL_summary.json")
    base = (BASELINE / live[0].name, BASELINE / live[1].name)
    present = lambda pair: all(p.exists() for p in pair)
    newest = lambda pair: datetime.fromtimestamp(max(p.stat().st_mtime for p in pair))
    read = lambda pair, label, final: (pd.read_csv(pair[0]), json.loads(pair[1].read_text(encoding="utf-8")), label, newest(pair), pair[0], final)
    final_stamp = newest(live) if present(live) else None
    prov_stamp = newest(prov) if present(prov) else None
    if final_stamp is not None and (prov_stamp is None or final_stamp > prov_stamp):
        return read(live, "FINAL: rebuilt twelve-month bill", True)
    if prov_stamp is not None:
        if final_stamp is not None:
            print("=" * 100)
            print(f"WARNING: PROVISIONAL bill selected although a final bill exists ({live[1].relative_to(ROOT)}, written "
                  f"{final_stamp:%Y-%m-%d %H:%M}) because the provisional files are newer ({prov_stamp:%Y-%m-%d %H:%M}). "
                  "Re-run the final bill or move the provisional files aside.")
            print("=" * 100)
        return read(prov, "PROVISIONAL: new fleet, missing months scaled", False)
    if present(base):
        return read(base, "PRE-REBUILD: saved baseline", False)
    raise FileNotFoundError("no bill components found")


components, summary, PROVENANCE, STAMP, SRC, IS_FINAL = load_bill()
components = annotate_groups(components)
components["ofgem_group"] = pd.Categorical(components["ofgem_group"], CATEGORY_ORDER, ordered=True)
# External lines: analyst placeholders (external_assumption) and the WS9 primary-source replacements
# (external_primary_source), which carry the value they replaced in reference_gbp_per_3100kwh_ex_vat.
components["external"] = components["source_class"].isin(["external_assumption", "external_primary_source"])
components["primary_source"] = components["source_class"].eq("external_primary_source")
if "reference_gbp_per_3100kwh_ex_vat" not in components.columns:
    components["reference_gbp_per_3100kwh_ex_vat"] = np.nan
IS_PROVISIONAL_NEW = (not IS_FINAL) and bool(summary.get("provisional"))
TAG = ("" if IS_FINAL else
       f"   [provisional: {len(summary['complete_months'])} months modelled, {len(summary['scaled_months'])} scaled]"
       if IS_PROVISIONAL_NEW else "   [pre-rebuild]")
by_category = components.groupby("ofgem_group", observed=True)[VALUE].sum().reindex(CATEGORY_ORDER)
EX_VAT, INC_VAT, VAT_RATE = float(summary["ex_vat_gbp"]), float(summary["total_gbp_inc_vat"]), float(summary["vat_rate"])
HH = 3.1 * float(summary["model_rebasing_factor"])                # MWh per household x 2024->2025 rebasing
DEMAND = float(summary["demand_mwh"])
# F18/F19: a bill line converts between household and system pounds through its own recovery entry
# (scripts/consumer_bill/line_recovery.py); a summary written before F18 falls back to its denominators.
from scripts.public_ownership import bill_recovery  # noqa: E402


def RATE(component_id):
    # GBP per household (report money) per system GBP (model money) of one bill line.
    return bill_recovery.household_per_system(summary, component_id)


CFD_RATE = RATE("cfd")                                   # the CfD line's recovery (eligible GSP volume, losses)
KEY_RATE = bill_recovery.flat_household_per_system(summary)   # flat on the Exchequer's key (metered supply)
line = lambda cid: float(components.loc[components.component_id.eq(cid), VALUE].iloc[0])
print(f"{PROVENANCE}  |  {SRC.relative_to(ROOT)}  (written {STAMP:%Y-%m-%d %H:%M})")
print(f"ex VAT {gbp(EX_VAT, 2)}   inc VAT {gbp(INC_VAT, 2)}   {len(components)} components")
""")

# =====================================================================================================
md(r"""
## Provenance
""")

code(r"""
def rebuilt(kind):
    files = {m: ANALYSIS.parent / "market" / f"{MONTH_PATTERN.format(month=m)}_{kind}.nc"
             for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
    return [m for m, p in files.items() if p.exists() and datetime.fromtimestamp(p.stat().st_mtime) > REBUILD_START]


lines = [f"BILL SOURCE   {PROVENANCE}",
         f"              {SRC.relative_to(ROOT)}  (written {STAMP:%Y-%m-%d %H:%M})",
         f"REBUILD       wholesale {len(rebuilt('wholesale'))}/12, balancing {len(rebuilt('balancing'))}/12 months solved since {REBUILD_START:%d %b %H:%M}",
         f"PRICE BASIS   {summary['currency']}; model year {summary['model_price_year']} rebased x{summary['model_rebasing_factor']:.4f}; VAT {VAT_RATE:.0%} once",
         ""]
if IS_FINAL:
    lines.append("All model-derived lines come from the twelve rebuilt monthly runs. No stand-in assumptions.")
elif IS_PROVISIONAL_NEW:
    dg = summary["diagnostics"]
    lines += [f"Modelled months ({len(summary['complete_months'])}): {', '.join(summary['complete_months'])}",
              f"Scaled months   ({len(summary['scaled_months'])}): {', '.join(summary['scaled_months'])}  "
              f"(pre-brownfield monthly values x price {dg['price_ratio_new_over_old_complete_months']:.3f}, "
              f"CfD {dg['cfd_ratio_new_over_old_complete_months']:.3f}, constraints {dg['constraints_ratio_new_over_old_complete_months']:.3f})",
              "", summary["provisional_note"], "",
              "Per-line basis:",
              *[f"  {cid:<38} {basis}" for cid, basis in components[["component_id", "provisional_basis"]].itertuples(index=False)
                if basis != "unchanged external input"],
              "",
              "Known biases: Hinkley Point C runs at 100% in balancing (zero-cost CfD offer), so its CfD line and the",
              "constraint saving are overstated; the scaled months include the winter peak, where scaling is least reliable.",
              "The public-ownership finance outputs and the hourly price series read the PRE-rebuild networks."]
else:
    lines += ["Pre-brownfield fleet: every CfD settles at one fallback strike per carrier, the RO comes from the Ofgem",
              "register, no nuclear unit is labelled CfD. CfD, RO and constraint lines will move when the rebuild lands."]
wrapped = []
for x in lines:
    wrapped += textwrap.wrap(x, 112, subsequent_indent=" " * (len(x) - len(x.lstrip()) + 4)) if len(x) > 112 else [x]
width = max(map(len, wrapped)) + 2
print("+" + "-" * width + "+")
for x in wrapped:
    print("| " + x.ljust(width - 1) + "|")
print("+" + "-" * width + "+")
""")

# =====================================================================================================
md(r"""
## 1. The bill
""")

code(r"""
fig = plt.figure(figsize=(11.5, 4.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.55)
ax0 = fig.add_subplot(gs[0]); ax0.axis("off"); ax0.set_ylim(0, 1)
ax0.text(0, 0.72, gbp(INC_VAT), fontsize=44, fontweight="bold", color=INK, va="center")
ax0.text(0, 0.22, f"per year, 3,100 kWh household, CP30 FFR 2030, including {gbp(INC_VAT - EX_VAT)} VAT at {VAT_RATE:.0%}"
         f"   |   {gbp(EX_VAT)} ex VAT", fontsize=10, color=INK_2, va="center")
ax0.text(0, -0.05, PROVENANCE, fontsize=8.5, color=INK_3, va="center", style="italic")

ax = fig.add_subplot(gs[1]); left = 0.0
for cat in CATEGORY_ORDER:
    v = by_category[cat]
    ax.barh(0, v, left=left, height=0.55, color=CATEGORY_COLOUR[cat], edgecolor=SURFACE, linewidth=2, zorder=3)
    if v / EX_VAT > 0.12:
        ax.text(left + v / 2, 0, f"{gbp(v)}\n{v / EX_VAT:.0%}", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold", zorder=4)
    else:
        ax.text(left + v / 2, 0.36, gbp(v), ha="center", va="bottom", fontsize=9, color=INK_2, zorder=4)
    ax.text(left + v / 2, -0.38, cat.replace(" & ", " &\n").replace(", ", ",\n"), ha="center", va="top", fontsize=8.5,
            color=CATEGORY_COLOUR[cat], fontweight="bold")
    left += v
ax.set_xlim(0, EX_VAT); ax.set_ylim(-1.05, 0.7); ax.set_yticks([]); ax.set_xticks([]); ax.grid(False)
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
title(ax, "Where the 2030 bill goes", f"GBP per year per 3,100 kWh household, ex VAT; grouped as in {CATEGORY_TITLE}")
plt.show()
""")

md(r"""
Solid bars are lines the model derives or informs; hatched bars are published external assumptions the model
never touches. The model-owned share says how far to trust the shape of the bill.
""")

code(r"""
rows = components[components[VALUE].abs() > 0.0005].sort_values(["ofgem_group", VALUE], ascending=[True, False]).reset_index(drop=True)
GAP, y, ypos, headers = 1.4, 0.0, [], []
for cat in CATEGORY_ORDER:
    sub = rows[rows["ofgem_group"].eq(cat)]
    if sub.empty:
        continue
    headers.append((y - 0.75, cat, by_category[cat]))
    for _ in range(len(sub)):
        ypos.append(y); y += 1.0
    y += GAP
rows["y"] = ypos

fig, ax = plt.subplots(figsize=(11.5, 0.36 * len(rows) + 2.6))
for _, r in rows.iterrows():
    c = CATEGORY_COLOUR[r["ofgem_group"]]
    if r["external"]:
        ax.barh(r["y"], r[VALUE], height=0.7, facecolor=c, alpha=0.28, edgecolor=c, hatch="////", linewidth=0.8, zorder=3)
    else:
        ax.barh(r["y"], r[VALUE], height=0.7, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.text(r[VALUE] + EX_VAT * 0.004, r["y"], f"{r[VALUE]:,.1f}", va="center", fontsize=8.5, color=INK_2)
for yh, cat, tot in headers:
    ax.text(-EX_VAT * 0.005, yh, f"{cat}   {gbp(tot)}", ha="right", va="center", fontsize=9.5, fontweight="bold", color=CATEGORY_COLOUR[cat])
ax.set_yticks(rows["y"]); ax.set_yticklabels(rows["component"], fontsize=8.5); ax.invert_yaxis(); ax.set_ylim(y - GAP + 0.6, -1.4)
ax.set_xlim(0, rows[VALUE].max() * 1.14); ax.xaxis.set_major_formatter(fmt_axis); ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False); ax.set_xlabel("GBP per year per 3,100 kWh household, ex VAT")
derived = components.loc[~components["external"], VALUE].sum()
primary = components.loc[components["primary_source"], VALUE].sum()
handles = [mpl.patches.Patch(facecolor=INK_3, label=f"model-derived or model-informed   {gbp(derived)}  ({derived / EX_VAT:.0%})"),
           mpl.patches.Patch(facecolor=INK_3, alpha=0.28, edgecolor=INK_3, hatch="////",
                             label=f"external   {gbp(EX_VAT - derived)}  ({1 - derived / EX_VAT:.0%}), of which primary-sourced (WS9) {gbp(primary)}")]
ax.legend(handles=handles, loc="lower right", handlelength=1.6, handleheight=1.2)
title(ax, "Bill components by Ofgem price-cap category", f"{CATEGORY_TITLE}. Hatched = external (official allowance or analyst placeholder); solid = derived from the CP30 runs")
plt.tight_layout(); plt.show()
""")

md(r"""
### Cross-check against an independent analyst forecast (electricitybills.uk)

Eleven external lines in this bill are still placeholders copied from the electricitybills.uk 2030 forecast, an
independent analyst site chosen because it was the only single published 2030 household stack on a 3,100 kWh
basis. The three largest (supplier costs and margin, DUoS, TNUoS) now come from primary sources (WS9 items 1 to 3:
Ofgem cap allowances, the Ofgem Annex 3 DUoS allowance cross-checked against RIIO-ED2, and NESO's five-year TNUoS
view), each keeping the value it replaced as a reference. The two bills therefore differ where the model replaces
the site's assumptions (wholesale price, procurement uplift, CfD, RO, balancing) and on those three lines. Their
2030 headline is only shown in a chart, so their total is reconstructed from their published components; their
FY 2025/26 outturn is the anchor.
""")

code(r"""
d = summary["diagnostics"]
EBUK_WHOLESALE_GBP_PER_MWH = 70.0                      # electricitybills.uk/2030: "Average wholesale price in 2030: £70/MWh", real 2025
EBUK_CFD_BN_2030_31 = 5.1                               # electricitybills.uk: CfD cost forecast £5.1bn in 2030/31
EBUK_TODAY = 880.70                                     # their FY 2025/26 bill, 3,100 kWh, from the methodology workbook
ebuk = {"Wholesale energy": EBUK_WHOLESALE_GBP_PER_MWH * 3.1, "Retail procurement uplift": float(d.get("procurement_bridge_reference_gbp_per_3100kwh", 52.0)),
        "BSUoS": float(d.get("bsuos_reference_gbp_per_3100kwh", 91.0)), "CfD": EBUK_CFD_BN_2030_31 * 1e9 * CFD_RATE / float(summary["model_rebasing_factor"]),
        "Renewables Obligation": float(d.get("ro_reference_gbp_per_3100kwh", 17.0))}
proc_ids = ["procurement_transaction_cost", "procurement_day_ahead_rehedging", "procurement_additional_risk_and_basis", "procurement_forward_premium", "procurement_losses"]
ours = {"Wholesale energy": line("direct_energy") + line("shaping_adjustment"), "Retail procurement uplift": sum(line(i) for i in proc_ids),
        "BSUoS": line("bsuos_constraints") + line("bsuos_other_services"), "CfD": line("cfd"), "Renewables Obligation": line("renewables_obligation")}
notes = {"Wholesale energy": f"theirs {EBUK_WHOLESALE_GBP_PER_MWH:.0f} GBP/MWh; ours {summary['wholesale_price_gbp_per_mwh']:.0f} demand-weighted (2024 real)",
         "Retail procurement uplift": "theirs a single allowance; ours Ofgem intensities on the modelled energy cost",
         "BSUoS": "theirs one NESO-forecast allowance; ours modelled redispatch and CfD curtailment compensation plus NESO non-constraint outturn",
         "CfD": f"theirs a levy forecast; ours settles every CfD generator against the strike table on modelled dispatch"
                + (f"; Hinkley Point C would add {float(d.get('nuclear_cfd_hpc_sensitivity_gbp', 0)) * CFD_RATE:,.0f} once labelled" if d.get("nuclear_cfd_hpc_sensitivity_gbp") else ""),
         "Renewables Obligation": f"theirs assumes the 75% relief continues; ours ends it in March 2029 per government guidance"
                                  + (f" (would be {d['ro_relief_case_values_gbp_per_3100kwh'].get('relief_extended', 0):,.0f} if extended)" if d.get("ro_relief_case_values_gbp_per_3100kwh") else "")}
# WS9: the primary-sourced lines are compared against the placeholder each replaced.
PRIMARY_LABEL = {"supplier_costs_and_margin": "Supplier costs and margin", "duos": "DUoS", "tnuos": "TNUoS"}
PRIMARY_NOTE = {"supplier_costs_and_margin": "theirs an analyst estimate; ours the sum of Ofgem's 2025 cap allowances (operating, cost to serve, debt, industry, payment, EBIT and headroom), real terms",
                "duos": "theirs an analyst estimate; ours Ofgem's Oct-Dec 2026 cap DUoS rolled forward on the RIIO-ED2 revenue path and ED2 RAV growth into ED3, with the ED3 years at Ofgem's SSMD allowed return (4.35% CPIH-real), real 2025 terms; low 125.37, high 144.73",
                "tnuos": "theirs NESO-derived; ours NESO's Sept 2025 five-year view for 2030/31 (domestic residual tariff plus locational charge) deflated at 2% CPIH"}
for cid, lab in PRIMARY_LABEL.items():
    row = components.loc[components.component_id.eq(cid)]
    if len(row) and row["primary_source"].iloc[0] and pd.notna(row["reference_gbp_per_3100kwh_ex_vat"].iloc[0]):
        ebuk[lab] = float(row["reference_gbp_per_3100kwh_ex_vat"].iloc[0]); ours[lab] = float(row[VALUE].iloc[0]); notes[lab] = PRIMARY_NOTE[cid]
placeholders = float(components.loc[components["source_class"].eq("external_assumption"), VALUE].sum())
externals = placeholders   # lines identical in both 2030 bills
ebuk_total = sum(ebuk.values()) + placeholders

fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
keys = list(ebuk); ys = np.arange(len(keys))
axl.barh(ys + 0.19, [ebuk[k] for k in keys], height=0.36, color=NEUTRAL, label="electricitybills.uk 2030", zorder=3)
axl.barh(ys - 0.19, [ours[k] for k in keys], height=0.36, color=SERIES[0], label="this model", zorder=3)
for yv, k in zip(ys, keys):
    axl.text(max(ebuk[k], ours[k]) + 4, yv + 0.19, gbp(ebuk[k]), va="center", fontsize=8.5, color=INK_3)
    axl.text(max(ebuk[k], ours[k]) + 4, yv - 0.19, f"{gbp(ours[k])}  ({ours[k] - ebuk[k]:+,.0f})", va="center", fontsize=8.5, color=INK_2, fontweight="bold")
    axl.text(max(ebuk[k], ours[k]) + 60, yv, textwrap.fill(notes[k], 46), va="center", fontsize=7.6, color=INK_3)
axl.set_yticks(ys); axl.set_yticklabels(keys, fontsize=9); axl.invert_yaxis(); axl.xaxis.set_major_formatter(fmt_axis)
axl.set_xlim(0, max(max(ebuk.values()), max(ours.values())) * 2.5); axl.grid(axis="y", visible=False)
axl.legend(loc="lower right", handlelength=1.4); title(axl, "Where the two bills differ", "the lines the model replaces plus the three WS9 primary-source lines; the other 11 lines are placeholders copied from the site")

totals = [("electricitybills.uk\nFY 2025/26 outturn", EBUK_TODAY, NEUTRAL), ("electricitybills.uk 2030\n(reconstructed)", ebuk_total * (1 + VAT_RATE), INK_3),
          ("this model 2030", INC_VAT, SERIES[0])]
for i, (lab, v, col) in enumerate(totals):
    axr.bar(i, v, width=0.62, color=col, edgecolor=SURFACE, zorder=3)
    axr.text(i, v + 12, gbp(v), ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
axr.set_xticks(range(3)); axr.set_xticklabels([t[0] for t in totals], fontsize=8.5); axr.tick_params(axis="x", length=0)
axr.set_ylim(0, max(t[1] for t in totals) * 1.15); axr.yaxis.set_major_formatter(fmt_axis); axr.grid(axis="x", visible=False)
title(axr, "Whole bill, inc VAT", "3,100 kWh household, real 2025")
caption(fig, "electricitybills.uk 2030 total = their published components plus 70 GBP/MWh wholesale, 52 procurement, 91 BSUoS, 17 RO and a 5.1 bn CfD levy "
        "spread over CfD eligible supply, plus 5% VAT.\nTheir FY 2025/26 figure is the workbook outturn for the same household. Placeholder lines identical in both 2030 bills: "
        f"{gbp(externals)} ex VAT.", y=-0.03)
plt.tight_layout(rect=(0, 0.04, 1, 0.94)); plt.show()
print(f"This model vs electricitybills.uk 2030 (reconstructed): {INC_VAT - ebuk_total * (1 + VAT_RATE):+,.0f} inc VAT per household; "
      f"vs their FY 2025/26 outturn: {INC_VAT - EBUK_TODAY:+,.0f}.")
""")

# =====================================================================================================
md(r"""
## 2. Public ownership: where the saving comes from, and what reaches the bill

Three financing cases from the finance layer (`resources/public_ownership/`), same run, same dispatch and prices.
**Private**: technology-specific private WACC from the CEPA 2025 hurdle rates, re-levered, after CfD-term and tax-shield
deductions (offshore wind 8.3%, solar 6.7%). **Public coordination**: 75 bp cheaper debt on the debt share and the
equity risk premium removed on a 30% state stake (offshore 7.5%). **Public ownership**: the real public borrowing rate
(4.8% nominal, 2.75% real) plus a construction premium of 0.5 to 1.5 points by risk rating (offshore 3.5%). Legacy
assets, existing CfD and ROC contracts and network ownership are unchanged in every case.

Four things drive the number, in order of size:

1. **The cost of capital roughly halves.** No equity risk premium (4.7 points), no credit spread over gilts, only a
   construction premium. Coordination moves the rate about one point, because the debt discount is 75 bp on the debt
   share and the premium comes off a 30% stake only.
2. **Capital-intensive assets turn a rate cut into a large cost cut.** With no fuel, annual cost is capex times an
   annuity factor; halving the rate on a 30-year asset cuts the factor by about 40%. O&M, fuel and carbon do not move,
   so the whole saving is capital recovery.
3. **Volume is offshore wind.** 36 GW of new offshore at 2,670 GBP/kW is 96 bn of capex; a 40% cut in its capital
   charge is 3.5 bn of the 6.0 bn.
4. **Route decides what reaches the bill.** Two thirds of the saving sits in CfD assets, where the strike is the
   contract and can be re-priced; a third sits in storage (no bill line) and merchant generation (reaches consumers
   only if the owner returns it).

The principle behind all four: cheaper finance changes the *cost* of the system; ownership changes *who keeps the
rent*. Only the first is a saving to the country.
""")

code(r"""
import yaml
CASES = ["public_control", "public_ownership"]
CASE_LABEL = {"private": "Private", "public_control": "Public control", "public_ownership": "Public ownership"}
CASE_COLOUR = {"public_control": "#4f8fd9", "public_ownership": "#0b4f6c"}
snap = yaml.safe_load((PUBLIC / "workbook_v5_assumptions.yaml").read_text(encoding="utf-8"))
costs = pd.read_csv(ROOT / "data" / "public_ownership" / "financing" / "technology_costs.csv").set_index("technology")
wacc = pd.DataFrame({row["technology"].replace("—", "-").replace("–", "-"): row["wacc"] for row in snap["technologies"]}).T
crf = lambda r, n: r * (1 + r) ** n / ((1 + r) ** n - 1)

tech_all = pd.read_csv(PUBLIC / f"{GROUP}_financing_savings_by_technology.csv")
new = tech_all[tech_all.finance_vintage.eq("new_build")]
by_tech = new.pivot_table(index="finance_technology", columns="case_id", values="annual_financing_saving_gbp", aggfunc="sum").fillna(0.0) / 1e9
cap_mw = new[new.case_id.eq("public_ownership")].groupby("finance_technology").capacity_mw.sum()
by_tech["capex_bn"] = cap_mw * costs.reindex(cap_mw.index)["capex_power_gbp_per_kw"] * 1e3 / 1e9
by_tech = by_tech.sort_values("public_ownership", ascending=True)
techs = [t for t in by_tech.index if t in wacc.index and t in costs.index]
life = costs.loc[techs, "economic_life_years"].astype(float)
factor_change = pd.Series({t: crf(wacc.loc[t, "public_ownership_new_build"], life[t]) / crf(wacc.loc[t, "private"], life[t]) - 1 for t in techs})

fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
ys = np.arange(len(techs))
for y, t in zip(ys, techs):
    pr, co, pu = wacc.loc[t, "private"], wacc.loc[t, "public_coordination"], wacc.loc[t, "public_ownership_new_build"]
    axl.plot([pu, pr], [y, y], color=SAVING, linewidth=3, alpha=0.35, zorder=2)
    axl.scatter(pr, y, s=110, color=NEUTRAL, edgecolor=INK_2, zorder=4); axl.scatter(co, y, s=90, color=CASE_COLOUR["public_control"], zorder=4)
    axl.scatter(pu, y, s=110, color=CASE_COLOUR["public_ownership"], zorder=5)
    axl.text(pr + 0.004, y, f"{pr:.1%} -> {pu:.1%}   capital charge {factor_change[t]:+.0%} over {life[t]:.0f} y", va="center", fontsize=8.5, color=INK_2)
axl.set_yticks(ys); axl.set_yticklabels(techs, fontsize=8.5); axl.set_xlim(0.02, 0.16)
axl.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0)); axl.grid(axis="y", visible=False)
axl.set_xlabel("WACC, real, post-2025 new build")
handles = [mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=9, color=NEUTRAL, markeredgecolor=INK_2, label="private"),
           mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=8, color=CASE_COLOUR["public_control"], label="public coordination"),
           mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=9, color=CASE_COLOUR["public_ownership"], label="public ownership")]
axl.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, fontsize=8.5)
title(axl, "Driver 1 and 2: the rate halves, the capital charge falls by a third or more", "workbook WACCs; annuity factor over each technology's economic life", tagged=False)

axr.barh(ys, by_tech.loc[techs, "public_ownership"], height=0.62, color=CASE_COLOUR["public_ownership"], label="public ownership", zorder=3)
axr.barh(ys, by_tech.loc[techs, "public_control"], height=0.62, color=CASE_COLOUR["public_control"], label="public coordination", zorder=4)
for y, t in zip(ys, techs):
    axr.text(by_tech.loc[t, "public_ownership"] + 0.05, y, f"{gbp(by_tech.loc[t, 'public_ownership'], 2)}bn  on {gbp(by_tech.loc[t, 'capex_bn'], 0)}bn of capex", va="center", fontsize=8.5, color=INK_2)
axr.set_yticks(ys); axr.set_yticklabels([""] * len(techs)); axr.set_xlim(0, by_tech["public_ownership"].max() * 1.9)
axr.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("£{x:,.1f}bn")); axr.grid(axis="y", visible=False)
axr.set_xlabel("annual financing saving vs private, GBP bn, 2024 real"); axr.legend(loc="lower right", fontsize=8.5)
title(axr, "Driver 3: volume, led by offshore wind", "post-2025 new build only; saving = capex x change in capital charge", tagged=False)
fig.suptitle("What drives the financing saving" + TAG, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.show()
""")

code(r"""
CASES = ["public_control", "public_ownership"]
CASE_LABEL = {"private": "Private", "public_control": "Public control", "public_ownership": "Public ownership"}
CASE_COLOUR = {"public_control": "#4f8fd9", "public_ownership": "#0b4f6c"}
tech = pd.read_csv(PUBLIC / f"{GROUP}_financing_savings_by_technology.csv")
fin = pd.read_csv(PUBLIC / f"{GROUP}_financing_summary.csv").set_index("case_id")
pool = lambda case, mask: float(tech.loc[tech.case_id.eq(case) & mask, "annual_financing_saving_gbp"].sum())
is_cfd = tech.finance_vintage.eq("new_build") & tech.support_type.astype(str).str.lower().eq("cfd")
is_beccs = tech.finance_technology.str.contains("CCUS", na=False)
is_storage = tech.finance_technology.str.contains("batter|LDES|storage", case=False, na=False)
saving = pd.DataFrame({c: {"total": float(fin.loc[c, "annual_financing_saving_vs_private_gbp"]),
                           "new CfD contracts": pool(c, is_cfd), "BECCS": pool(c, is_beccs), "storage": pool(c, is_storage)} for c in CASES}).T
saving["merchant and other generation"] = saving["total"] - saving["new CfD contracts"] - saving["BECCS"] - saving["storage"]

delta = pd.DataFrame(0.0, index=components.component_id, columns=CASES)      # routed change per household, ex VAT
BY_CASE = PUBLIC / f"{GROUP}_consumer_bill_by_ownership_case_summary.json"
ROUTES = PUBLIC / f"{GROUP}_financing_saving_routes.csv"
ADJUSTMENTS = PUBLIC / f"{GROUP}_cfd_strike_adjustments.csv"
if BY_CASE.exists() and ROUTES.exists():
    # WP-C: the routed change is the live projector's own CfD line per ownership case (rule
    # project_public_ownership_consumer_bill), not a subtraction from the finance bridge.
    by_case = json.loads(BY_CASE.read_text(encoding="utf-8"))["cases"]
    routes = pd.read_csv(ROUTES)
    # Release 2 lines the per-case bill carries beside the CfD line: the LDES cap-and-floor top-up (in
    # every case, private included), the state merchant portfolio at cost (W1) and its shortfall levy.
    NEW_LINES = ["ldes_cap_and_floor", "public_cfd_at_cost", "public_cost_recovery_contracting", "public_surplus_rebate", "public_portfolio_shortfall_levy"]
    for line in NEW_LINES:
        if line not in delta.index:
            delta.loc[line] = 0.0
    private_new = by_case["private"].get("new_lines_gbp_per_3100kwh_ex_vat", {})
    for c in CASES:
        delta.loc["cfd", c] = by_case[c]["cfd_line_delta_gbp_per_3100kwh_ex_vat"]
        for line in NEW_LINES:
            delta.loc[line, c] = by_case[c].get("new_lines_gbp_per_3100kwh_ex_vat", {}).get(line, 0.0) - private_new.get(line, 0.0)
        # OW10: Sizewell C's RAB return at the public rate, carried as an adjustment line and shown on the line it adjusts.
        if "sizewell_c_rab" in delta.index:
            delta.loc["sizewell_c_rab", c] = (by_case[c].get("new_lines_gbp_per_3100kwh_ex_vat", {}).get("sizewell_c_rab_public_rate_adjustment", 0.0)
                                              - private_new.get("sizewell_c_rab_public_rate_adjustment", 0.0))
    PRIVATE_TOTAL = float(by_case["private"]["total_gbp_inc_vat"])          # forecast bill plus the private LDES floor line
    r1 = routes[routes.route.eq("R1_cfd_repricing_bill")].set_index("case_id")
    unrouted = pd.Series({c: (saving.loc[c, "total"] - float(r1.loc[c, "saving_economic_life_gbp_2024"])
                              - float(by_case[c].get("bill_allocated_ldes_gbp_2024", 0.0))) * CFD_RATE for c in CASES})
    ROUTE_METHOD = pd.read_csv(ADJUSTMENTS)["method"].iloc[0] if ADJUSTMENTS.exists() else "contract_vintage"
    ROUTE_SOURCE = "live projector, rule project_public_ownership_consumer_bill"
    for c in CASES:
        e = by_case[c]
        print(f"{CASE_LABEL[c]}: gross {e['gross_total_gbp_inc_vat']:,.0f} ({e['delta_gross_vs_private_gbp_inc_vat']:+,.1f}), "
              f"net {e['net_total_gbp_inc_vat']:,.0f} ({e['delta_net_vs_private_gbp_inc_vat']:+,.1f}) inc VAT; rule {e.get('recycling_rule')}; "
              f"retained in public account {e.get('public_account_retained_gbp_2024', 0) / 1e9:.2f}bn; "
              f"Capacity Market sensitivity {e.get('capacity_market_sensitivity_gbp_per_3100kwh_ex_vat') or 0:.1f} per household")
else:
    # Fallback while the per-case bill is not on disk: the superseded meeting-figures subtraction.
    for c in CASES:
        delta.loc["cfd", c] = -saving.loc[c, "new CfD contracts"] * CFD_RATE
        delta.loc["allocated_ccs", c] = -min(saving.loc[c, "BECCS"] * RATE("allocated_ccs"), line("allocated_ccs"))
    unrouted = (saving["storage"] + saving["merchant and other generation"]) * CFD_RATE
    ROUTE_METHOD = "commissioning_vintage;economic_life (bridge subtraction, superseded)"
    ROUTE_SOURCE = "finance bridge subtraction (per-case bill not built)"
    PRIVATE_TOTAL = INC_VAT
routed = delta.sum()
print(f"Routed change from {ROUTE_SOURCE}; eligibility and term: {ROUTE_METHOD}")

# ---- figure: where the financing saving sits, and how much of it has a route --------------------------
fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
parts = [("new CfD contracts", SERIES[2], False), ("BECCS", "#7fd3b3", False), ("storage", NEUTRAL, True), ("merchant and other generation", "#d9d7d0", True)]
for i, c in enumerate(CASES):
    bottom = 0.0
    for name, col, hatched in parts:
        v = saving.loc[c, name] / 1e9
        axl.bar(i, v, bottom=bottom, width=0.6, facecolor=col if not hatched else "none", edgecolor=INK_3 if hatched else SURFACE,
                hatch="////" if hatched else None, linewidth=1.0, zorder=3, label=name if i == 0 else None)
        if v > 0.12:
            axl.text(i, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=INK if hatched else "white", fontweight="bold")
        bottom += v
    axl.text(i, bottom + 0.1, f"{gbp(bottom, 2)}bn", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
axl.set_xticks(range(len(CASES))); axl.set_xticklabels([CASE_LABEL[c] for c in CASES], fontsize=9.5); axl.tick_params(axis="x", length=0)
axl.set_ylim(0, saving["total"].max() / 1e9 * 1.25); axl.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("£{x:,.0f}bn")); axl.grid(axis="x", visible=False)
axl.legend(loc="upper left", fontsize=8.5, handlelength=1.4)
title(axl, "Annual financing saving vs private", "GBP bn a year, 2024 real; hatched = no bill route yet", tagged=False)
xs = np.arange(len(CASES))
axr.bar(xs - 0.2, routed.values, width=0.38, color=[CASE_COLOUR[c] for c in CASES], edgecolor=SURFACE, zorder=3, label="routed today (CfD, CCS)")
axr.bar(xs + 0.2, -unrouted.values, width=0.38, facecolor="none", edgecolor=INK_3, hatch="////", linewidth=1.0, zorder=3, label="not yet routed, at full pass-through (scale only)")
for x, c in zip(xs, CASES):
    axr.text(x - 0.2, routed[c] - 1.2, f"{routed[c]:+,.0f}", ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    axr.text(x + 0.2, -unrouted[c] - 1.2, f"up to {-unrouted[c]:,.0f}", ha="center", va="top", fontsize=9, color=INK_3)
axr.axhline(0, color=INK_3, linewidth=0.8); axr.set_xticks(xs); axr.set_xticklabels([CASE_LABEL[c] for c in CASES], fontsize=9.5); axr.tick_params(axis="x", length=0)
axr.set_ylim(min(routed.min(), -unrouted.max()) * 1.35, 6); axr.yaxis.set_major_formatter(fmt_axis); axr.grid(axis="x", visible=False)
axr.legend(loc="lower left", fontsize=8.5, handlelength=1.4); title(axr, "Change per household", "GBP per year ex VAT, 2025 real", tagged=False)
fig.suptitle("Where the financing saving sits, and how much of it reaches the bill today" + TAG, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.show()
""")

code(r"""
STEP_LABEL = [("cfd", "New CfD\ncontracts\nre-priced"), ("ldes_cap_and_floor", "LDES cap\nand floor"),
              ("public_cfd_at_cost", "State\nrenewables\non CfD at cost"), ("public_cost_recovery_contracting", "Merchant\nportfolio\nat cost (W1)"),
              ("public_surplus_rebate", "Surplus\nrebate"),
              ("public_portfolio_shortfall_levy", "Shortfall\nlevy"), ("allocated_ccs", "CCS levy\n(BECCS)"),
              ("sizewell_c_rab", "Sizewell C\nRAB at the\npublic rate")]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
for ax, c in zip(axes, CASES):
    steps = [("Private\nbill", PRIVATE_TOTAL, "total")]
    steps += [(label, float(delta.loc[line, c]), "step") for line, label in STEP_LABEL if line in delta.index and abs(delta.loc[line, c]) > 0.005]
    steps += [("VAT on\nchanges", routed[c] * VAT_RATE, "step"),
              (CASE_LABEL[c].replace(" ", "\n") + "\nbill", PRIVATE_TOTAL + routed[c] * (1 + VAT_RATE), "total")]
    running = 0.0
    for i, (label, v, kind) in enumerate(steps):
        if kind == "total":
            ax.bar(i, v, width=0.62, color=CASE_COLOUR[c] if i else NEUTRAL, edgecolor=SURFACE, zorder=3)
            ax.text(i, v + PRIVATE_TOTAL * 0.012, gbp(v), ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK); running = v
        else:
            ax.bar(i, v, bottom=running, width=0.62, color=SAVING if v < 0 else COST, edgecolor=SURFACE, zorder=3)
            ax.text(i, running + v - PRIVATE_TOTAL * 0.012, f"{v:+,.1f}", ha="center", va="top", fontsize=9.5, color=INK_2); running += v
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=8.5)
    ax.set_ylim(PRIVATE_TOTAL * 0.85, PRIVATE_TOTAL * 1.05); ax.yaxis.set_major_formatter(fmt_axis); ax.grid(axis="x", visible=False); ax.tick_params(axis="x", length=0)
    title(ax, CASE_LABEL[c], f"routed saving {gbp(-routed[c] * (1 + VAT_RATE))} inc VAT per household", tagged=False)
axes[0].set_ylabel("GBP per year per household, inc VAT (axis truncated)")
fig.suptitle("From the private bill to each public case" + TAG, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
caption(fig, f"CfD re-pricing rule: {ROUTE_METHOD}. Under contract vintage only rounds awarded from 2025 (AR7, AR7a and unassigned capacity) are re-priced; "
             "earlier-awarded, later-commissioned capacity keeps its strike and its saving sits in the public account (route R2).")
plt.tight_layout(rect=(0, 0.02, 1, 0.93)); plt.show()
""")

# =====================================================================================================
md(r"""
## 3. Impact map: every line, every case, and how big it could get

One row per bill line, grouped by what each cost pays for (the functional view, not Ofgem's categories). The two coloured columns show the status of each line under the two public
cases and the quantified change where one exists. The right-hand column is our judgment of the largest plausible effect
under full public ownership, with the reasoning beside it; it is not a model output, and lines marked T are transfers
between consumers, taxpayers and owners rather than savings to the country.
""")

code(r"""
# Each row: (status, largest plausible effect under public ownership in GBP per household, is it a transfer, note).
# Notes follow one pattern: what would happen, what limits it, where it stands.
REGISTER = {
    "direct_energy": ("identified", -4, False,
        "The clearing price does not move. Selling the state's merchant plant at cost, or rebating its surplus, saves about 4 (section 4). The scarcity mark-up is modelled inside the wholesale price (item MR3) and is small in 2030."),
    "hedging": ("identified", -1, False,
        "Supplier hedging and risk costs could fall if a public generator sold fixed-price forward contracts. Not modelled."),
    "procurement_losses": ("none", 0, False, "Physical losses; no ownership channel."),
    "capacity_market": ("identified", -12, False,
        "The auction price is set by the cheapest new plant's cost of capital and paid to every winner, so public batteries and gas would pull it down. The 2030/31 T-4 is modelled (56 GBP/kW, range 36 to 69, item MR4); a public bidder at its own cost of capital is not."),
    "bsuos_constraints": ("identified", -16, True,
        "Consumers pay it through BSUoS. If the state owned the curtailed wind it could waive the payment to itself: a bill saving, but the state absorbs the lost revenue. Cost-based public bids are solved as P2 with state bids at cost (item BM1); better siting is not modelled."),
    "bsuos_other_services": ("identified", -5, False,
        "NESO (already public) buys frequency response, reserve and reactive power from batteries, gas and demand-side providers. The payments are credited to the providing plant (OW16), so state plant receives them; supplying them at cost is not modelled."),
    "bsuos_voltage_and_stability_constraints": ("identified", -1, False,
        "Voltage and stability services, credited to the providing plant like the other services (OW16); part goes to synchronous condensers outside the register."),
    "cfd": ("quantified", None, False,
        "New contracts re-priced to the public cost of capital: quantified above. Existing contracts unchanged, including Hinkley Point C (2016 CfD, about 18 per household once fully settled; a transfer if acquired)."),
    "renewables_obligation": ("held", -30, True,
        "Closed scheme on older, largely depreciated plant; unchanged. If the state bought the stations at book value and cancelled the certificates, the rent would move from today's owners to consumers; at market value it is paid up front in the price. Ends 2037."),
    "feed_in_tariffs": ("none", 0, False, "Closed scheme, levelised per MWh of supply (DESNZ projection, item F1); no channel."),
    "warm_home_discount": ("none", 0, False, "Social policy, recovered per kWh from domestic customers since April 2026 (OBR 2030-31, item F1); no channel."),
    "eco": ("none", 0, False, "Off bills from April 2026 (item F1)."),
    "aahedc": ("held", 0, False, "Follows distribution charges."),
    "sizewell_c_rab": ("quantified", None, False,
        "The levy is mostly a return on capital during construction. In the public cases its return is scaled to the public rate on the state's share (OW10; the whole levy treated as return, an upper bound). The station is not in the 2030 network."),
    "nuclear_contract_counterparty": ("none", 0, False, "Administrative cost; no channel."),
    "allocated_ccs": ("quantified", None, False,
        "BECCS at Drax on a Power BECCS CfD (electricity leg only; strike 109.90 in 2012 prices, a declared assumption since no contract is signed). A post-2025 contract, so it is re-priced to the public cost of capital with the CfD line (item MR7); quantified above."),
    "remaining_ccs": ("none", 0, False, "Net Zero Teesside Power's DPA. The 742 MW gas CCS plant is in the model (a committed overlay dispatched ahead of unabated gas, and a register row whose support revenue is this line); the payment is an upper bound from the ECC business case, not settled from dispatch (item F1). No channel."),
    "bics_levy": ("held", -1, False, "British Industry Supercharger: other consumers fund a 60% network-charge discount for energy-intensive industry, so it follows network charges."),
    "tnuos": ("held", -21, False,
        "Return on transmission assets at 2.75% instead of 5.6%, no corporation-tax allowance, and the interconnector cap-and-floor pass-through is a separate line. The wires are out of the base cases; the networks lever (debt guarantee, acquisition) prices this as a sensitivity."),
    "duos": ("held", -29, False,
        "Same for the 14 distribution companies; the ED2 price control model gives the shares. Out of the base cases; a networks-lever sensitivity."),
    "interconnector_cap_and_floor_pass_through": ("held", 0, False,
        "Interconnector cap-and-floor revenue passed back through TNUoS; follows the networks case."),
    "ldes_cap_and_floor": ("quantified", None, False,
        "The LDES floor top-up, in every case (item D4(c)); each public case carries its own top-up, at the public cost of capital where the state owns the plant."),
    "battery_revenue_top_up": ("none", 0, False,
        "Tops private batteries up to their required revenue after ancillary services and the sub-hourly uplift (OW16). The same line in every case; the state's own batteries' share returns through its surplus."),
    "smart_meter_net_cost_change": ("none", 0, False, "No channel."),
    "supplier_costs_and_margin": ("identified", -5, False,
        "Margin and headroom are set as a share of costs, so they shrink a little with every other saving. Public retail is out of scope."),
}
EXTRA = []
HEDGING_IDS = ["shaping_adjustment", "procurement_transaction_cost", "procurement_day_ahead_rehedging", "procurement_additional_risk_and_basis", "procurement_forward_premium"]
STATUS_COLOUR = {"quantified": SAVING, "identified": "#eda100", "held": NEUTRAL, "none": SURFACE}
STATUS_LABEL = {"quantified": "quantified: routed to the bill above", "identified": "identified: real channel, no route or data yet",
                "held": "held unchanged by decision (legacy assets, contracts, networks)", "none": "no channel"}
NONE = ("none", 0, False, "")

# Rows grouped by what each cost pays for; the five hedging lines collapse into one row.
rows = []
for cat in FUNCTIONAL_ORDER:
    sub = components[components.functional_group.eq(cat) & (components[VALUE].abs() > 0.0005)].sort_values(VALUE, ascending=False)
    if sub.empty:
        continue
    rows.append(("header", cat, None, None))
    hedged = False
    for cid, name, val in sub[["component_id", "component", VALUE]].itertuples(index=False):
        if cid in HEDGING_IDS:
            if not hedged:
                total = float(components.loc[components.component_id.isin(HEDGING_IDS), VALUE].sum())
                rows.append(("row", "Hedging, shaping and procurement allowances", total, REGISTER["hedging"])); hedged = True
            continue
        rows.append(("row", name, val, REGISTER.get(cid, NONE) + (cid,)))
if EXTRA:
    rows.append(("header", "Lines public ownership would create", None, None))
for name, st, mx, tr, note in EXTRA:
    rows.append(("row", name, None, (st, mx, tr, note)))

fig, ax = plt.subplots(figsize=(15.5, 0.48 * len(rows) + 2.6))
y = 0.0
for kind, name, val, meta in rows:
    if kind == "header":
        ax.text(-0.02, y + 0.5, name, ha="right", va="center", fontsize=9.5, fontweight="bold", color=INK)
        ax.plot([0, 3.7], [y + 0.5, y + 0.5], color=GRID, linewidth=1.0, zorder=1)
        y += 0.8; continue
    st, mx, transfer, note = meta[0], meta[1], meta[2], meta[3]
    cid = meta[4] if len(meta) > 4 else None
    if mx is None and cid:
        mx = delta.loc[cid, "public_ownership"]
    for j, c in enumerate(CASES):
        ax.add_patch(mpl.patches.Rectangle((j, y), 1, 1, facecolor=STATUS_COLOUR[st], edgecolor=SURFACE, linewidth=2.5, zorder=2))
        if st == "quantified" and cid:
            ax.text(j + 0.5, y + 0.5, f"{delta.loc[cid, c]:+,.1f}", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=3)
        elif st == "none":
            ax.text(j + 0.5, y + 0.5, "–", ha="center", va="center", fontsize=9, color=INK_3, zorder=3)
    shade = min(abs(mx) / 30, 1.0) * 0.55 if mx else 0
    ax.add_patch(mpl.patches.Rectangle((2.15, y), 0.75, 1, facecolor=SAVING if mx < 0 else SURFACE, alpha=shade if mx else 1, edgecolor=SURFACE, linewidth=2.5, zorder=2))
    ax.text(2.525, y + 0.5, (f"{mx:+,.0f}" + (" T" if transfer else "")) if mx else "–", ha="center", va="center", fontsize=9.5,
            fontweight="bold" if mx else "normal", color=INK if mx else INK_3, zorder=3)
    ax.text(3.0, y + 0.5, textwrap.fill(note, 88), va="center", fontsize=8, color=INK_2 if note else INK_3, linespacing=1.15)
    ax.text(-0.02, y + 0.5, f"{name}   {val:,.0f}" if val is not None else name, ha="right", va="center", fontsize=8.5, color=INK_2)
    y += 1.0
ax.set_xlim(0, 3.0); ax.set_ylim(y, -0.9)
ax.set_xticks([0.5, 1.5, 2.525]); ax.set_xticklabels(["Public coordination\nchange today", "Public ownership\nchange today", "Largest plausible\n(ownership)"],
                                                   fontsize=9, fontweight="bold")
ax.xaxis.tick_top(); ax.tick_params(axis="both", length=0); ax.set_yticks([])
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
handles = [mpl.patches.Patch(facecolor=STATUS_COLOUR[k], edgecolor=INK_3 if k == "none" else SURFACE, label=v) for k, v in STATUS_LABEL.items()]
handles.append(mpl.patches.Patch(facecolor=SURFACE, edgecolor=SURFACE, label="T = transfer: consumers pay less only because the state or a private owner receives less; the system costs no less to run"))
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.005), ncol=3, handlelength=1.4, handleheight=1.1, fontsize=8.5)
ax.set_title("How each bill line can move under public ownership" + TAG, pad=62, color=INK)
ax.annotate("Row label: bill line and its value today, GBP per household ex VAT.  Coloured cells: status, with the quantified change where one exists "
            "(negative = saving).\nRight column: our judgment of the largest plausible effect under full public ownership, with the reasoning alongside; "
            "not a model output.",
            xy=(0, 1), xycoords="axes fraction", xytext=(0, 32), textcoords="offset points", fontsize=8.5, color=INK_3, va="bottom", ha="left", annotation_clip=False)
plt.subplots_adjust(left=0.23, right=0.58, top=0.90, bottom=0.07); plt.show()

maxes = [(m[1] if m[1] is not None else delta.loc[m[4], "public_ownership"], m[2]) for k, n, v, m in rows if k == "row"]
total_max = sum(v for v, _ in maxes); transfers = sum(v for v, t in maxes if t)
print(f"Routed today under public ownership: {routed['public_ownership']:+,.0f} ex VAT.  Sum of largest plausible effects: {total_max:+,.0f} ex VAT "
      f"({-total_max / EX_VAT:.0%} of the bill), of which {transfers:+,.0f} are transfers rather than savings to the country, and "
      f"{REGISTER['tnuos'][1] + REGISTER['duos'][1]:+,.0f} is the networks sensitivity, out of the base cases. Coordination is roughly a fifth of ownership.")

# component table for reference, written not displayed
table = components[["ofgem_group", "functional_group", "component", "component_id", VALUE, "source_class"] + (["provisional_basis"] if "provisional_basis" in components.columns else []) + ["source"]].copy()
lookup = lambda k: REGISTER.get("hedging" if k in HEDGING_IDS else k, NONE)
table["ownership_status"] = table.component_id.map(lambda k: STATUS_LABEL[lookup(k)[0]])
for c in CASES:
    table[f"delta_{c}_GBP"] = table.component_id.map(delta[c]).round(2)
table["max_plausible_ownership_GBP"] = table.component_id.map(lambda k: lookup(k)[1] if lookup(k)[1] is not None else round(delta.loc[k, "public_ownership"], 1))
table["is_transfer"] = table.component_id.map(lambda k: lookup(k)[2])
table["ownership_note"] = table.component_id.map(lambda k: lookup(k)[3])
out = ANALYSIS / f"{CASE}_bill_figure_table.csv"; table.drop(columns="component_id").to_csv(out, index=False)
print(f"Component table written to {out.relative_to(ROOT)}")
""")

# =====================================================================================================
md(r"""
### Is the financing framing limiting? Ownership as an enabler of reform

Deliberately, yes. The cases isolate the financing effect on a fixed physical system, which is the part the workbook's
evidence supports, so the 6.0 bn should be read as the **financing floor**, not the ownership effect. Channels the
framing excludes, none in the model, roughly sized where possible:

- **Spillover to private cost of capital.** A credible public anchor investor, offtaker or guarantor lowers perceived
  policy risk for every project, not only the state's own; this is the real case for coordination beyond a 30% stake.
  Each point off the private WACC on post-2025 build is worth about 1.3 bn a year, roughly double on the whole fleet.
  The largest unmodelled upside.
- **Long-term contracting and a clean power pool.** A state portfolio selling at cost is a cost-based contract for most
  new clean output; at scale a split market or green pool becomes simple, with consumers paying average clean cost
  rather than a gas-set marginal price. For CfD assets the strike already does this; for merchant assets it is the
  cost-recovery route (section 4); for the legacy fleet it needs acquisition or contract reform.
- **Coordinated planning of generation, network and storage** against 3.8 bn of constraint costs; **adequacy without an
  auction** (up to about 12 GBP per household); **curtailment redesign** (27 GBP of compensation becomes intra-public);
  **auction dynamics** (already inside the CfD re-pricing); **retail and hedging** (a few GBP, outside the workbook scope).

And the channels that cut the other way: the saving is financed by borrowing (the workbook's Assumption B raises gilt
yields with debt-to-GDP by about 0.2 points by 2030); corporation tax and VAT receipts fall; construction and volume
risk move to taxpayers with only the construction premium priced; and build pace is held fixed in every case.

**How to say it.** The model shows the financing effect on new build: 6 bn a year, of which 44 GBP per household has
a bill route today and about 190 could plausibly be reached across all channels. Ownership as an enabler of market
reform is where the larger prize may sit, and also where the evidence is thinnest and the downside risks live.
""")

md(r"""
## 4. Public cost recovery, and why pay-as-bid is a ceiling not a case

Everything above holds the wholesale line fixed. This section recalculates it for **generation** under three
settlement rules on the same dispatch and hourly prices, so the only thing that changes is who keeps the inframarginal
rent (the gap between what a generator is paid and what it needs).

- **Pay-as-clear** (today): every generator is paid the clearing price; every generator keeps its rent.
- **Public cost recovery**: the same, except the state's own assets are paid what they cost (fuel, carbon, and capital
  and operating cost at the public rate); the state returns its rent, private generators keep theirs.
- **Pay-as-bid with truthful bids**: each merchant generator is paid its marginal cost; nobody keeps rent.

`rent = market revenue - fuel and carbon - required revenue at the case WACC`. A negative rent is missing money and
*raises* the wholesale cost. CfD cohorts are paid the strike whatever the market does, so for them cost recovery means
lowering the strike to cost, which is section 2; they are shown for comparison, not double-counted. Storage is
excluded: the model gives it arbitrage revenue only, and its route is the cap-and-floor line. Pay-as-bid is a ceiling:
REMA found bids drift to the expected clearing price, removing rent removes the capital-recovery signal (the Capacity
Market would pick up the bill), and ownership does not confer a settlement rule.

Dispatch is the balancing stage of the twelve monthly networks (about ninety seconds on first run, cached afterwards).
Prices are the pre-rebuild hourly series, the only one on disk, so the wholesale base here is that series' line.
""")

code(r"""
SETTLE_CACHE = PUBLIC / f"{GROUP}_cohort_settlement.csv"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NET_FILES = [ROOT / "resources" / "market" / f"{MONTH_PATTERN.format(month=m)}_balancing.nc" for m in MONTHS]
EXCLUDE = {"load_shedding", "EU_market"}      # unserved energy at VOLL; external import position
# Consumer flexibility is demand, not an asset that earns (F5, 27 Oct 2026).
from scripts.demand.demand_carriers import DEMAND_FLEX_STORAGE_CARRIERS  # noqa: E402
EXCLUDE |= set(DEMAND_FLEX_STORAGE_CARRIERS)
hourly = pd.read_csv(ANALYSIS / f"{CASE}_consumer_bill_future_hourly_inputs.csv", parse_dates=["snapshot"]).set_index("snapshot")
price = hourly["price_gbp_per_mwh"]
BASE_WHOLESALE = float((price * hourly.demand_mw).sum() / hourly.demand_mw.sum()) * HH     # wholesale line at this price series


def settle_networks():
    import pypsa
    rows = []
    for path in [p for p in NET_FILES if p.exists()]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); n = pypsa.Network(str(path))
        pi = price.reindex(n.snapshots).ffill()
        for kind, static, p in (("Generator", n.generators, n.generators_t.p), ("StorageUnit", n.storage_units, n.storage_units_t.p)):
            p = p.reindex(columns=static.index, fill_value=0.0)
            support = static.get("support_type", pd.Series("", index=static.index)).fillna("").replace("", "unclassified")
            # ROC generators bid below zero because certificate income is folded into the bid: a subsidy, not a cost.
            mc = static["marginal_cost"].reindex(static.index).fillna(0.0).clip(lower=0.0)
            frame = pd.DataFrame({"component_type": kind, "carrier": static["carrier"].values, "support_type": support.values,
                                  "dispatch_mwh": p.clip(lower=0).sum().values, "market_revenue_gbp": p.mul(pi, axis=0).sum().values,
                                  "fuel_carbon_gbp": p.clip(lower=0).mul(mc, axis=1).sum().values}, index=static.index)
            frame["month"] = MONTHS[NET_FILES.index(path)]
            frame["network_written"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes")
            rows.append(frame[~frame.carrier.isin(EXCLUDE)])
    out = pd.concat(rows).groupby(["component_type", "carrier", "support_type"], as_index=False).agg(
        dispatch_mwh=("dispatch_mwh", "sum"), market_revenue_gbp=("market_revenue_gbp", "sum"), fuel_carbon_gbp=("fuel_carbon_gbp", "sum"),
        months=("month", "nunique"), newest_network=("network_written", "max"), oldest_network=("network_written", "min"))
    out.to_csv(SETTLE_CACHE, index=False)
    return out


# Recompute only when all twelve networks are on disk and one is newer than the cache; while the rebuild
# is deleting and rewriting months, fall back to the cached settlement.
present = [p for p in NET_FILES if p.exists()]
newest_net = max(datetime.fromtimestamp(p.stat().st_mtime) for p in present)
cache_fresh = SETTLE_CACHE.exists() and datetime.fromtimestamp(SETTLE_CACHE.stat().st_mtime) > newest_net
if SETTLE_CACHE.exists() and (cache_fresh or len(present) < len(NET_FILES)):
    settlement = pd.read_csv(SETTLE_CACHE); src = "cache" + ("" if cache_fresh else f" ({len(NET_FILES) - len(present)} months mid-rebuild)")
else:
    settlement = settle_networks(); src = f"{len(present)} balancing networks"
settlement["capture_price"] = settlement.market_revenue_gbp / settlement.dispatch_mwh.replace(0, np.nan)
settlement["fuel_carbon_per_mwh"] = settlement.fuel_carbon_gbp / settlement.dispatch_mwh.replace(0, np.nan)

from scripts.public_ownership.register_io import read_register
# Only the cases and columns used below: the register holds every case (about 1,000 from 30 Sep 2026).
register = read_register(PUBLIC / f"{GROUP}_asset_finance_register.parquet", cases=["private"] + CASES,
                         columns=["case_id", "evidence_status", "support_type", "component_type", "carrier",
                                  "finance_vintage", "finance_technology", "annual_required_revenue_gbp",
                                  "annual_fixed_om_gbp", "dispatch_mwh", "ownership_share"])
register = register[register.evidence_status.eq("resolved")].copy()
register["support_type"] = register["support_type"].fillna("unclassified").replace("", "unclassified")
cohort = (register.groupby(["case_id", "component_type", "carrier", "support_type", "finance_vintage", "finance_technology"], as_index=False)
          .agg(required_revenue_gbp=("annual_required_revenue_gbp", "sum"), fixed_om_gbp=("annual_fixed_om_gbp", "sum"), dispatch_mwh=("dispatch_mwh", "sum"), ownership_share=("ownership_share", "max"))
          .merge(settlement[["component_type", "carrier", "support_type", "capture_price", "fuel_carbon_per_mwh"]], on=["component_type", "carrier", "support_type"], how="left"))
cohort["market_revenue_gbp"] = cohort.dispatch_mwh * cohort.capture_price
cohort["fuel_carbon_gbp"] = cohort.dispatch_mwh * cohort.fuel_carbon_per_mwh.fillna(0.0)
cohort["rent_gbp"] = cohort.market_revenue_gbp - cohort.fuel_carbon_gbp - cohort.required_revenue_gbp
cohort["public_rent_gbp"] = cohort.ownership_share * cohort.rent_gbp
to_hh = lambda gbp_2024: gbp_2024 * RATE("direct_energy")        # wholesale route: the wholesale line's own recovery

generation = cohort[cohort.component_type.eq("Generator")]
merchant_new = generation[generation.finance_vintage.eq("new_build") & ~generation.support_type.str.lower().eq("cfd")]
cfd_new = generation[generation.finance_vintage.eq("new_build") & generation.support_type.str.lower().eq("cfd")]
w1_total = merchant_new[merchant_new.case_id.isin(CASES)].groupby("case_id").public_rent_gbp.sum() / 1e9
merchant_gen = settlement[settlement.component_type.eq("Generator") & ~settlement.support_type.str.lower().eq("cfd")]
pab_rent_gbp = float((merchant_gen.market_revenue_gbp - merchant_gen.fuel_carbon_gbp).sum())

print(f"Settlement from {src}; networks written {settlement.oldest_network.min()[:10]} to {settlement.newest_network.max()[:10]}.")
print(f"Clearing price (2024 real): mean {price.mean():.0f}, median {price.median():.0f}, demand-weighted {BASE_WHOLESALE / HH:.0f} GBP/MWh; "
      f"{(price <= 1).mean():.0%} of hours at or below 1, {(price >= 100).mean():.0%} at or above 100.")
for c in CASES:
    sub = merchant_new[merchant_new.case_id.eq(c)]
    print(f"{CASE_LABEL[c]}: new merchant generation rent {gbp(sub.rent_gbp.sum() / 1e9, 2)}bn at case WACC; state share {gbp(w1_total[c], 2)}bn = "
          f"{-to_hh(w1_total[c] * 1e9):+.1f} per household on the wholesale line ({'a saving' if w1_total[c] > 0 else 'a cost: missing money'}).")
print(f"Pay-as-bid ceiling (truthful bids, all merchant generators): {gbp(pab_rent_gbp / 1e9, 2)}bn = {to_hh(pab_rent_gbp):.0f} per household against a wholesale line of {gbp(BASE_WHOLESALE)}.")
""")

md(r"""
**How to read the next figure.** Grey marker: what the cohort is paid today (market capture price for merchant,
strike for CfD). Hollow ring: what it costs at the private WACC. Filled dot: what it costs at the public WACC. Green
means the public cost is below what it is paid, so selling at cost would save consumers money; red means it is above.
The figure after it applies the merchant result to the wholesale line and adds the CfD route from section 2.
""")

code(r"""
def strike_by_carrier():
    # The strike table the CfD line was settled at; the provisional bill stores it differently, so fall back to
    # the pre-rebuild summary, which matches the hourly price series used here.
    for cand in (summary, json.loads((ANALYSIS / f"{CASE}_consumer_bill_future_summary.json").read_text(encoding="utf-8"))):
        rows = (cand.get("cfd_strike_provenance") or {}).get("by_strike")
        if rows:
            return {row["carrier"]: float(row["strike_price"]) for row in rows}
    return {}


STRIKE = strike_by_carrier()
tech_saving = tech[tech.case_id.eq("public_ownership") & tech.finance_vintage.eq("new_build")].set_index(["finance_technology", "support_type"])["annual_financing_saving_gbp"]


def unit_economics(frame, kind):
    # Per MWh: what the state-owned cohort is paid, and what it costs at private and public WACC.
    own = frame[frame.case_id.eq("public_ownership")].groupby("finance_technology").agg(
        carrier=("carrier", "first"), dispatch_mwh=("dispatch_mwh", "sum"), market_revenue_gbp=("market_revenue_gbp", "sum"),
        fuel_carbon_gbp=("fuel_carbon_gbp", "sum"), required_revenue_gbp=("required_revenue_gbp", "sum"), rent_gbp=("rent_gbp", "sum"))
    priv = frame[frame.case_id.eq("private")].groupby("finance_technology").required_revenue_gbp.sum()
    out = pd.DataFrame({"kind": kind, "paid": (own.market_revenue_gbp / own.dispatch_mwh) if kind == "merchant" else own.carrier.map(STRIKE),
                        "cost_private": (own.fuel_carbon_gbp + priv.reindex(own.index)) / own.dispatch_mwh,
                        "cost_public": (own.fuel_carbon_gbp + own.required_revenue_gbp) / own.dispatch_mwh})
    out["household_gbp"] = (-own.rent_gbp * RATE("direct_energy") if kind == "merchant"
                            else -pd.Series({t: tech_saving.get((t, "CfD"), 0.0) for t in own.index}) * CFD_RATE)
    return out.dropna(subset=["paid"])


unit = pd.concat([unit_economics(merchant_new, "merchant"), unit_economics(cfd_new, "CfD")]).sort_values(["kind", "cost_public"], ascending=[False, False])
merch = unit[unit.kind.eq("merchant")]

fig, ax = plt.subplots(figsize=(13.5, 0.5 * len(unit) + 1.6))
ylab = np.arange(len(unit))
for y, (name, r) in zip(ylab, unit.iterrows()):
    worse = r.cost_public > r.paid
    ax.plot([r.paid, r.cost_public], [y, y], color=COST if worse else SAVING, linewidth=3, alpha=0.35, zorder=2)
    ax.scatter(r.paid, y, s=110, marker="s" if r.kind == "merchant" else "D", color=NEUTRAL, edgecolor=INK_2, zorder=4)
    ax.scatter(r.cost_private, y, s=90, facecolor="none", edgecolor=INK_3, linewidth=1.3, zorder=4)
    ax.scatter(r.cost_public, y, s=110, color=COST if worse else SAVING, zorder=5)
    ax.text(max(r.cost_public, r.cost_private, r.paid) * 1.3, y,
            f"{'market pays' if r.kind == 'merchant' else 'strike'} £{r.paid:,.0f}, costs £{r.cost_public:,.0f}  ->  {r.household_gbp:+.1f}/household "
            f"({'wholesale line' if r.kind == 'merchant' else 'CfD line, s.2'})", va="center", fontsize=8.5, color=INK_2)
ax.axhline(len(merch) - 0.5, color=GRID, linewidth=1.2)
ax.set_xscale("log"); ax.set_xlim(1, 3e5); ax.set_yticks(ylab)
ax.set_yticklabels([f"{t} ({k})" for t, k in zip(unit.index, unit.kind)], fontsize=8.5); ax.invert_yaxis()
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"£{v:,.0f}")); ax.grid(axis="y", visible=False)
ax.set_xlabel("GBP per MWh of its own output, 2024 real (log scale)")
handles = [mpl.lines.Line2D([], [], marker="s", linestyle="", markersize=9, color=NEUTRAL, markeredgecolor=INK_2, label="paid: market capture price (merchant)"),
           mpl.lines.Line2D([], [], marker="D", linestyle="", markersize=8, color=NEUTRAL, markeredgecolor=INK_2, label="paid: CfD strike"),
           mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="none", markeredgecolor=INK_3, label="cost at private WACC"),
           mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=9, color=COST, label="cost at public WACC, above what it is paid"),
           mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=9, color=SAVING, label="cost at public WACC, below what it is paid")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, fontsize=8.5)
title(ax, "State-owned new build: what it is paid vs what it costs, per MWh",
      "colour: cost at public WACC vs what it is paid today; household figure: change from private to public cost")
caption(fig, "Generation only; storage excluded. CfD 'paid' is the fallback strike now in the bill; the private-WACC cost of offshore wind exceeds it, "
        "which is why section 2 lowers strikes by the cost change rather than to cost.", y=-0.01)
plt.tight_layout(rect=(0, 0.01, 1, 0.95)); plt.show()
""")

code(r"""
fig, (axr, axt) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})

above, below = merch.household_gbp[merch.household_gbp > 0], merch.household_gbp[merch.household_gbp <= 0]
pieces = [("Pay-as-\nclear", BASE_WHOLESALE, "total"), ("Cost\nabove\nmarket", float(above.sum()), "step"),
          ("Cost\nbelow\nmarket", float(below.sum()), "step"), ("Cost\nrecovery", None, "total"), ("Pay-as-\nbid\nceiling", BASE_WHOLESALE - to_hh(pab_rent_gbp), "bound")]
running = BASE_WHOLESALE
for i, (lab, v, kind) in enumerate(pieces):
    if kind == "total":
        v = running if v is None else v
        axr.bar(i, v, width=0.62, color=CASE_COLOUR["public_ownership"] if i else NEUTRAL, edgecolor=SURFACE, zorder=3)
        axr.text(i, v + BASE_WHOLESALE * 0.02, gbp(v) + ("" if not i else f"  ({v - BASE_WHOLESALE:+,.0f})"), ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
        running = v
    elif kind == "step":
        axr.bar(i, v, bottom=running, width=0.62, color=COST if v > 0 else SAVING, edgecolor=SURFACE, zorder=3)
        axr.text(i, running + max(v, 0) + BASE_WHOLESALE * 0.02, f"{v:+,.0f}", ha="center", va="bottom", fontsize=9.5, color=INK_2); running += v
    else:
        axr.bar(i, v, width=0.62, facecolor="none", edgecolor=INK_3, hatch="////", linewidth=1.0, zorder=3)
        axr.text(i, v + BASE_WHOLESALE * 0.02, f"{gbp(v)}  ({v - BASE_WHOLESALE:+,.0f})", ha="center", va="bottom", fontsize=9.5, color=INK_3)
axr.axvline(3.5, color=GRID, linewidth=1.2)
axr.set_xticks(range(len(pieces))); axr.set_xticklabels([p[0] for p in pieces], fontsize=8.5); axr.tick_params(axis="x", length=0)
axr.set_ylim(0, BASE_WHOLESALE * 1.3); axr.yaxis.set_major_formatter(fmt_axis); axr.grid(axis="x", visible=False)
axr.set_ylabel("GBP per household per year")
title(axr, "Wholesale line, merchant generation", f"public ownership; coordination (30% share) lands at {gbp(BASE_WHOLESALE - to_hh(w1_total['public_control'] * 1e9))} "
      f"({-to_hh(w1_total['public_control'] * 1e9):+,.0f})", tagged=False)

combined = pd.DataFrame({c: {"CfD line (strike to cost, s.2)": float(delta.loc["cfd", c]), "Wholesale line (merchant at cost)": -to_hh(w1_total[c] * 1e9)} for c in CASES}).T
xs = np.arange(len(CASES)); bottom_neg = np.zeros(len(CASES)); bottom_pos = np.zeros(len(CASES))
for col, colour in zip(combined.columns, [SERIES[2], SERIES[0]]):
    vals = combined[col].values; neg = np.where(vals < 0, vals, 0.0); pos = np.where(vals > 0, vals, 0.0)
    axt.bar(xs, neg, bottom=bottom_neg, width=0.6, color=colour, edgecolor=SURFACE, label=col, zorder=3); bottom_neg += neg
    axt.bar(xs, pos, bottom=bottom_pos, width=0.6, color=colour, edgecolor=SURFACE, zorder=3); bottom_pos += pos
for x, c in zip(xs, CASES):
    axt.text(x, min(bottom_neg[x], 0) - 1.5, f"{float(combined.loc[c].sum()):+,.0f} ex VAT", ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
axt.axhline(0, color=INK_3, linewidth=0.8)
axt.set_xticks(xs); axt.set_xticklabels([CASE_LABEL[c] for c in CASES], fontsize=9.5); axt.tick_params(axis="x", length=0)
axt.yaxis.set_major_formatter(fmt_axis); axt.grid(axis="x", visible=False)
axt.set_ylim(min(bottom_neg.min(), 0) * 1.3 - 5, max(bottom_pos.max(), 0) + 8); axt.legend(loc="lower left", fontsize=8.5)
title(axt, "Both cost-recovery routes together", "change per household vs private, ex VAT", tagged=False)

net = float(merch.household_gbp.sum()); tot_own = float(combined.loc["public_ownership"].sum())
verdict = f"lowers the wholesale line by {gbp(-net)}" if net < 0 else f"raises the wholesale line by {gbp(net)}"
fig.suptitle(f"Merchant generation at cost {verdict} per household; with CfD strikes at cost the total is {tot_own:+,.0f} under public ownership" + TAG,
             x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
caption(fig, "Pay-as-bid ceiling: every merchant generator paid its marginal cost with no capital recovered; REMA found bids would not stay there, "
        "and the Capacity Market would have to fund the missing capital.")
plt.tight_layout(rect=(0, 0.02, 1, 0.93)); plt.show()
""")

md(r"""
## 5. What buying the legacy fleet would unlock

Everything above leaves the legacy fleet alone: its CfD and ROC contracts stay, its owners keep the rent, and the two
largest "T" rows in the impact map (curtailment compensation and the Renewables Obligation) stay transfers because the
state does not own the plant. Acquisition changes that. It is also the mechanism behind most of the workbook's
Scenario 3 headline, and the piece a colleague is now costing.

The framing that fits the rest of this notebook: **acquisition buys the legacy fleet's cash flows.** Today consumers
pay for that fleet three ways: the market price on its output, CfD top-ups on its supported output, and the RO levy.
If the state bought the fleet it would receive those flows, and could either keep the contracts and recycle the
surplus or cancel its own contracts and put only the residual on bills. Either way the consumer effect is the same:

```
consumer saving = cash flows the state now receives - public financing cost of the purchase
cash flows      = legacy net operating cash (market revenue - fuel, carbon and fixed O&M)
                  + legacy CfD top-ups (including Hinkley Point C) + ROC income
financing cost  = purchase price x annuity factor at the real public borrowing rate over the term
```

The purchase price decides whether this is a saving to consumers or a transfer they pay for up front: at book value
the rent moves from today's owners to consumers; at market value the same rent is capitalised into the price. The
cell below uses the workbook's book-value placeholder (75 bn renewables, 25 bn nuclear and gas) and shows the
result across a range of prices and terms. **Replace the placeholder with the acquisition-cost work when it lands**,
ideally by station or technology so the streams and the price can be matched cohort by cohort.
""")

code(r"""
# ---- inputs the acquisition-cost work will replace --------------------------------------------------
ACQUISITION_PRICE_BN = None            # set to the costed figure (GBP bn, 2024 real); None uses the workbook placeholder
WORKBOOK_BOOK_VALUE_BN = float(snap["assumptions"].get("generation_book_value_renewables", {}).get("value", 75.0) or 75.0) + 25.0
PUBLIC_REAL_RATE = float(snap["assumptions"]["public_borrowing_real"]["value"])
TERMS = (15, 25)                        # financing terms: legacy contracts run to the mid-2030s; 25 y is the long bookend
price_bn = ACQUISITION_PRICE_BN if ACQUISITION_PRICE_BN is not None else WORKBOOK_BOOK_VALUE_BN
price_label = "costed acquisition price" if ACQUISITION_PRICE_BN is not None else "workbook book-value placeholder"

# ---- the legacy fleet's cash flows today, 2024 real GBP bn ---------------------------------------------
legacy = generation[generation.finance_vintage.eq("legacy") & generation.case_id.eq("private")].copy()
legacy["net_cash_gbp"] = legacy.market_revenue_gbp - legacy.fuel_carbon_gbp - legacy.fixed_om_gbp
cfd_leg = legacy[legacy.support_type.str.lower().eq("cfd")]
topups = float(sum(max(STRIKE.get(r.carrier, 0.0) - (r.capture_price if pd.notna(r.capture_price) else 0.0), 0.0) * r.dispatch_mwh
                   for r in cfd_leg.itertuples()))
hpc = float(summary["diagnostics"].get("nuclear_cfd_settled_payment_gbp")
            or summary["diagnostics"].get("nuclear_cfd_hpc_sensitivity_gbp") or 0.0)  # settled HPC CfD once labelled; sensitivity only before
ro = summary.get("ro_scheme_cost", {})
roc_income = (float(ro["roc_volume"]) * float(ro["notional_roc_value_gbp_per_roc"]) / float(summary["model_rebasing_factor"])
              if ro.get("roc_volume") else line("renewables_obligation") / RATE("renewables_obligation"))
streams = pd.Series({
    "net operating cash, merchant and unlabelled": float(legacy.loc[~legacy.support_type.str.lower().isin(["cfd", "roc"]), "net_cash_gbp"].sum()) / 1e9,
    "net operating cash, CfD and ROC stations": float(legacy.loc[legacy.support_type.str.lower().isin(["cfd", "roc"]), "net_cash_gbp"].sum()) / 1e9,
    "legacy CfD top-ups (fallback strikes)": topups / 1e9,
    "Hinkley Point C top-up (diagnostic)": hpc / 1e9,
    "ROC income": roc_income / 1e9,
})
total_streams = float(streams.sum())
annuity = lambda r, n: r * (1 + r) ** n / ((1 + r) ** n - 1)
per_hh = lambda bn: bn * 1e9 * KEY_RATE          # flat on the Exchequer's key (metered supply)

fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.3]})
cols = [NEUTRAL, "#d9d7d0", SERIES[2], "#7fd3b3", SERIES[3]]
bottom = 0.0
for (name, v), col in zip(streams.items(), cols):
    axl.bar(0, v, bottom=bottom, width=0.55, color=col, edgecolor=SURFACE, label=name, zorder=3)
    if abs(v) > 0.35:
        axl.text(0, bottom + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=9, color=INK, fontweight="bold")
    bottom += v
axl.text(0, bottom + 0.2, f"{gbp(total_streams, 1)}bn a year", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
for i, n in enumerate(TERMS):
    cost = price_bn * annuity(PUBLIC_REAL_RATE, n)
    axl.bar(1 + i, cost, width=0.55, color=CASE_COLOUR["public_ownership"], alpha=0.55 + 0.35 * i, edgecolor=SURFACE, zorder=3)
    axl.text(1 + i, cost + 0.2, f"{gbp(cost, 1)}bn", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
axl.set_xticks([0, 1, 2]); axl.set_xticklabels(["Cash flows the state\nwould receive"] + [f"Financing cost,\n{gbp(price_bn)}bn over {n} y" for n in TERMS], fontsize=8.5)
axl.tick_params(axis="x", length=0); axl.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("£{x:,.0f}bn")); axl.grid(axis="x", visible=False)
axl.set_ylim(0, max(total_streams, price_bn * annuity(PUBLIC_REAL_RATE, TERMS[0])) * 1.25)
axl.legend(loc="upper right", fontsize=8, handlelength=1.4)
title(axl, "Legacy fleet: what the state would buy, and what the purchase would cost", f"GBP bn a year, 2024 real; {price_label}; public rate {PUBLIC_REAL_RATE:.2%} real", tagged=False)

prices = np.linspace(0, 250, 251)
for i, n in enumerate(TERMS):
    saving_hh = np.array([per_hh(total_streams - p * annuity(PUBLIC_REAL_RATE, n)) for p in prices])
    axr.plot(prices, saving_hh, color=CASE_COLOUR["public_ownership"], alpha=0.55 + 0.35 * i, linewidth=2.2, label=f"financed over {n} years")
    be = total_streams / annuity(PUBLIC_REAL_RATE, n)
    axr.scatter([be], [0], color=CASE_COLOUR["public_ownership"], alpha=0.55 + 0.35 * i, zorder=5)
    axr.text(be, 3, f"breakeven {gbp(be)}bn", ha="center", va="bottom", fontsize=8.5, color=INK_2)
axr.axhline(0, color=INK_3, linewidth=0.8); axr.axvline(price_bn, color=INK_3, linewidth=0.8, linestyle="--")
axr.text(price_bn + 2, axr.get_ylim()[0] * 0.9 if axr.get_ylim()[0] < 0 else 5, price_label, fontsize=8.5, color=INK_3, va="bottom")
axr.fill_between(prices, 0, axr.get_ylim()[1], color=SAVING, alpha=0.05); axr.fill_between(prices, axr.get_ylim()[0], 0, color=COST, alpha=0.05)
axr.set_xlim(0, 250); axr.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("£{x:,.0f}bn")); axr.yaxis.set_major_formatter(fmt_axis)
axr.set_xlabel("acquisition price for the legacy fleet, GBP bn, 2024 real"); axr.set_ylabel("consumer saving per household, GBP per year ex VAT")
axr.legend(loc="upper right", fontsize=8.5)
title(axr, "Consumer saving against the price paid", "cash flows minus financing cost, allocated over CfD eligible supply; above zero = consumers gain", tagged=False)
fig.suptitle("Acquisition buys the legacy fleet's cash flows; the price decides whether consumers gain" + TAG, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
caption(fig, "Streams use the pre-rebuild settlement and the CfD fallback strikes, which understate the older, higher legacy strikes; Hinkley Point C is the bill diagnostic. "
        "Legacy CfDs and the RO expire between now and 2037, so the\nstreams decline: the 15-year term is the honest bookend. Fixed O&M is the register's estimate. "
        "Purchase price, remaining lives and any premium over book value are the acquisition-cost work's inputs.", y=-0.03)
plt.tight_layout(rect=(0, 0.04, 1, 0.93)); plt.show()

for n in TERMS:
    cost = price_bn * annuity(PUBLIC_REAL_RATE, n)
    print(f"At {gbp(price_bn)}bn over {n} years: financing {gbp(cost, 1)}bn a year against {gbp(total_streams, 1)}bn of cash flows -> "
          f"{per_hh(total_streams - cost):+,.0f} per household ex VAT; breakeven price {gbp(total_streams / annuity(PUBLIC_REAL_RATE, n))}bn.")
print("What acquisition unlocks in the map above: the RO row (-30 T) and the curtailment-compensation row (-16 T) stop being transfers the state "
      "pays for and become consumer savings funded by the purchase; the legacy refinancing saving is the gap between private and public capital "
      "charges on the acquired book value, which needs the colleague's remaining-life and book-value figures rather than the register's as-new costs.")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
for _cell in cells:
    if _cell.cell_type == "code":
        _cell.source = _cell.source.replace("__REPORT_CASE__", _ARGS.case).replace("__REPORT_GROUP__", _ARGS.group).replace("__REPORT_PATTERN__", _ARGS.pattern)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
