"""
Generate a per-scenario Jupyter notebook for two-stage market dispatch analysis.

This script is called by the Snakemake rule 'generate_market_analysis_notebook'.
It programmatically builds a self-contained .ipynb that mirrors the interactive
analysis in notebooks/market_analysis.ipynb, but is parameterised for a single
scenario so it can be executed as part of the automated workflow.

Input files (provided via snakemake.input):
  - wholesale_dispatch_csv, wholesale_storage_csv, wholesale_links_csv,
    wholesale_price_csv  — Stage 1 (copperplate) outputs
  - balancing_dispatch_csv, redispatch_summary_csv, constraint_costs_csv,
    congestion_csv, price_comparison_csv — Stage 2 (BM) outputs
  - wholesale_network, balancing_network — solved .nc network files

Output:
  - {scenario}_market_notebook.ipynb saved to resources/analysis/
"""

import json
import logging
import os
from pathlib import Path

# ── Logging ─────────────────────────────────────────────────────────────────
try:
    log_file = snakemake.log[0]
except (NameError, AttributeError, IndexError, TypeError):
    log_file = "generate_market_notebook.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


# ── Notebook builder ─────────────────────────────────────────────────────────

def build_notebook(scenario: str, inputs: dict) -> dict:
    """Return a complete nbformat 4 notebook for the given scenario."""

    cells = []

    # ── Title ────────────────────────────────────────────────────────────────
    cells.append(_md(
        f"# PyPSA-GB Market Dispatch Analysis — `{scenario}`\n"
        "\n"
        "Two-stage wholesale + balancing mechanism (BM) market dispatch results.\n"
        "\n"
        "**Two-stage structure:**\n"
        "1. **Stage 1 (Wholesale):** Copperplate dispatch — all AC thermal limits relaxed "
        "to 1 TWh. Produces a uniform system marginal price.\n"
        "2. **Stage 2 (Balancing Mechanism):** Full network constraints reimposed. "
        "Generators are anchored to their wholesale positions; objective = minimise "
        "cost of redispatch (bid/offer prices).\n"
        "\n"
        "*Generated automatically by the PyPSA-GB Snakemake workflow.*"
    ))

    # ── Cell 1: imports + config ─────────────────────────────────────────────
    # Embed relative paths; notebook will resolve them at runtime so
    # the generated file remains portable across different user directories.
    rel_paths = {}
    for key, path_val in inputs.items():
        # store path relative to workspace root if possible
        rel = os.path.relpath(path_val, start=Path.cwd())
        rel_paths[key] = rel.replace('\\', '/')  # use forward slashes for notebook

    cells.append(_code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.ticker as mticker\n"
        "import warnings\n"
        "import os\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# ── Scenario & file paths ────────────────────────────────────────────\n"
        f"SCENARIO = {repr(scenario)}\n"
        "\n"
        "# relative paths from repository root; will be made absolute below\n"
        "PATHS = {\n"
        f"    'wholesale_dispatch':  {repr(rel_paths['wholesale_dispatch_csv'])},\n"
        f"    'wholesale_storage':   {repr(rel_paths['wholesale_storage_csv'])},\n"
        f"    'wholesale_links':     {repr(rel_paths['wholesale_links_csv'])},\n"
        f"    'wholesale_price':     {repr(rel_paths['wholesale_price_csv'])},\n"
        f"    'balancing_dispatch':  {repr(rel_paths['balancing_dispatch_csv'])},\n"
        f"    'redispatch':          {repr(rel_paths['redispatch_summary_csv'])},\n"
        f"    'costs':               {repr(rel_paths['constraint_costs_csv'])},\n"
        f"    'congestion':          {repr(rel_paths['congestion_csv'])},\n"
        f"    'price_cmp':           {repr(rel_paths['price_comparison_csv'])},\n"
        f"    'wholesale_network':   {repr(rel_paths['wholesale_network'])},\n"
        f"    'balancing_network':   {repr(rel_paths['balancing_network'])},\n"
        "}\n"
        "\n"
        "# convert to absolute by searching upward for repo root at runtime\n"
        "cwd = Path(os.getcwd())\n"
        "def _find_base(path):\n"
        "    # look for directory containing resources/market\n"
        "    if (path / 'resources' / 'market').exists():\n"
        "        return path\n"
        "    for parent in path.parents:\n"
        "        if (parent / 'resources' / 'market').exists():\n"
        "            return parent\n"
        "    return path\n"
        "base = _find_base(cwd)\n"
        "for k,v in list(PATHS.items()):\n"
        "    p = Path(v)\n"
        "    if not p.is_absolute():\n"
        "        candidate = base / v\n"
        "        if candidate.exists():\n"
        "            PATHS[k] = str(candidate.resolve())\n"
        "        else:\n"
        "            PATHS[k] = str((cwd / v).resolve())\n"
        "\n"
        "COLORS = {\n"
        "    'wind_offshore':                  '#1f77b4',\n"
        "    'wind_onshore':                   '#aec7e8',\n"
        "    'solar_pv':                       '#ffdd57',\n"
        "    'nuclear':                        '#9467bd',\n"
        "    'CCGT':                           '#e86414',\n"
        "    'Battery':                        '#2ca02c',\n"
        "    'Pumped Storage Hydroelectricity':'#17becf',\n"
        "    'large_hydro':                    '#98df8a',\n"
        "    'load_shedding':                  '#d62728',\n"
        "    'OCGT':                           '#fd8d3c',\n"
        "    'waste_to_energy':                '#8c564b',\n"
        "    'landfill_gas':                   '#bcbd22',\n"
        "    'biogas':                         '#6baed6',\n"
        "    'marine':                         '#3182bd',\n"
        "    'LAES':                           '#74c476',\n"
        "    'Domestic Battery':               '#7fc97f',\n"
        "}\n"
        "\n"
        "def fmt_gbp(x, pos=None):\n"
        "    if abs(x) >= 1e6: return f'\\u00a3{x/1e6:.1f}M'\n"
        "    if abs(x) >= 1e3: return f'\\u00a3{x/1e3:.0f}k'\n"
        "    return f'\\u00a3{x:.0f}'\n"
        "\n"
        "def fmt_mwh(x, pos=None):\n"
        "    if abs(x) >= 1e6: return f'{x/1e6:.1f} TWh'\n"
        "    if abs(x) >= 1e3: return f'{x/1e3:.0f} GWh'\n"
        "    return f'{x:.0f} MWh'\n"
        "\n"
        "print('Libraries loaded.')"
    ))

    # ── Cell 2: load CSVs ────────────────────────────────────────────────────
    cells.append(_code(
        "# ── Load all market CSVs ─────────────────────────────────────────────\n"
        "wp  = pd.read_csv(PATHS['wholesale_price'],    index_col=0, parse_dates=True)\n"
        "gen = pd.read_csv(PATHS['wholesale_dispatch'], index_col=0, parse_dates=True)\n"
        "su  = pd.read_csv(PATHS['wholesale_storage'],  index_col=0, parse_dates=True)\n"
        "lnk = pd.read_csv(PATHS['wholesale_links'],    index_col=0, parse_dates=True)\n"
        "bdp = pd.read_csv(PATHS['balancing_dispatch'], index_col=0, parse_dates=True)\n"
        "rd  = pd.read_csv(PATHS['redispatch'])\n"
        "cc  = pd.read_csv(PATHS['costs'])\n"
        "cg  = pd.read_csv(PATHS['congestion'])\n"
        "pc  = pd.read_csv(PATHS['price_cmp'],          index_col=0, parse_dates=True)\n"
        "\n"
        "print(f'Loaded {SCENARIO}:')\n"
        "print(f'  Timesteps       : {len(wp)}')\n"
        "print(f'  Gen assets (WS) : {len(gen.columns)}')\n"
        "print(f'  Storage (WS)    : {len(su.columns)}')\n"
        "print(f'  BM assets       : {len(rd)}')\n"
        "print(f'  Congested comps : {len(cg)}')"
    ))

    # ── Section 1: Wholesale dispatch overview ────────────────────────────────
    cells.append(_md("## 1. Wholesale Dispatch Overview"))

    cells.append(_code(
        "import pypsa\n"
        "\n"
        "def dispatch_by_carrier(dispatch_df, network_path):\n"
        "    \"\"\"Sum dispatch by carrier, using network for carrier lookup.\"\"\"\n"
        "    try:\n"
        "        n = pypsa.Network(network_path)\n"
        "        carriers = n.generators['carrier'].rename_axis('name')\n"
        "        total = dispatch_df.sum().rename('MWh').to_frame()\n"
        "        total['carrier'] = carriers.reindex(total.index)\n"
        "        return total.groupby('carrier')['MWh'].sum()\n"
        "    except Exception:\n"
        "        return dispatch_df.sum().sum()\n"
        "\n"
        "def storage_by_carrier(storage_df, network_path):\n"
        "    try:\n"
        "        n = pypsa.Network(network_path)\n"
        "        carriers = n.storage_units['carrier'].rename_axis('name')\n"
        "        total = storage_df.sum().rename('MWh').to_frame()\n"
        "        total['carrier'] = carriers.reindex(total.index)\n"
        "        return total.groupby('carrier')['MWh'].sum()\n"
        "    except Exception:\n"
        "        return storage_df.sum().sum()\n"
        "\n"
        "gen_by_c = dispatch_by_carrier(gen, PATHS['wholesale_network'])\n"
        "su_by_c  = storage_by_carrier(su,  PATHS['wholesale_network'])\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "fig.suptitle(f'Wholesale Dispatch — Total by Carrier ({SCENARIO})', fontsize=13)\n"
        "\n"
        "# Generators\n"
        "ax = axes[0]\n"
        "if isinstance(gen_by_c, pd.Series):\n"
        "    plot = gen_by_c[(gen_by_c.abs() > 100) & (gen_by_c.index != 'load_shedding')]\n"
        "    plot = plot.sort_values(ascending=False)\n"
        "    plot.div(1000).plot.bar(ax=ax,\n"
        "        color=[COLORS.get(c, '#999') for c in plot.index],\n"
        "        edgecolor='white', linewidth=0.5)\n"
        "    ax.set_title('Generation by Carrier')\n"
        "    ax.set_ylabel('Energy (GWh)')\n"
        "    ax.tick_params(axis='x', labelrotation=45)\n"
        "    ax.xaxis.set_tick_params(labelsize=8)\n"
        "\n"
        "# Storage\n"
        "ax2 = axes[1]\n"
        "if isinstance(su_by_c, pd.Series):\n"
        "    su_plot = su_by_c[su_by_c.abs() > 10].sort_values(ascending=False)\n"
        "    su_plot.div(1000).plot.bar(ax=ax2,\n"
        "        color=[COLORS.get(c, '#999') for c in su_plot.index],\n"
        "        edgecolor='white', linewidth=0.5)\n"
        "    ax2.set_title('Storage Dispatch by Carrier')\n"
        "    ax2.set_ylabel('Energy (GWh)')\n"
        "    ax2.tick_params(axis='x', labelrotation=45)\n"
        "    ax2.xaxis.set_tick_params(labelsize=8)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Section 2: Wholesale price ────────────────────────────────────────────
    cells.append(_md("## 2. Wholesale Price"))

    cells.append(_code(
        "fig, axes = plt.subplots(2, 1, figsize=(12, 7))\n"
        "fig.suptitle(f'Wholesale Price — {SCENARIO}', fontsize=13)\n"
        "\n"
        "# Price time series\n"
        "ax = axes[0]\n"
        "ax.plot(wp.index, wp['wholesale_price'], marker='o', color='#1f77b4')\n"
        "ax.axhline(0, color='black', linewidth=0.8, linestyle='--')\n"
        "ax.set_title('Wholesale Clearing Price')\n"
        "ax.set_ylabel('\\u00a3/MWh')\n"
        "ax.tick_params(axis='x', labelrotation=30)\n"
        "\n"
        "# Price spread diagnostic\n"
        "ax2 = axes[1]\n"
        "spread_color = '#d62728' if wp['price_spread'].max() > 10 else '#2ca02c'\n"
        "ax2.plot(wp.index, wp['price_spread'], marker='s', color=spread_color)\n"
        "ax2.set_title('Price Spread across Buses (copperplate diagnostic — should be ~0)')\n"
        "ax2.set_ylabel('\\u00a3/MWh spread')\n"
        "ax2.tick_params(axis='x', labelrotation=30)\n"
        "max_spread = wp['price_spread'].max()\n"
        "ax2.text(0.02, 0.95, f'Max: \\u00a3{max_spread:,.0f}/MWh',\n"
        "         transform=ax2.transAxes, va='top', color=spread_color, fontsize=9)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print(f'Wholesale price  mean : \\u00a3{wp[\"wholesale_price\"].mean():.2f}/MWh')\n"
        "print(f'Wholesale price  min  : \\u00a3{wp[\"wholesale_price\"].min():.2f}/MWh')\n"
        "print(f'Wholesale price  max  : \\u00a3{wp[\"wholesale_price\"].max():.2f}/MWh')\n"
        "print(f'Copperplate spread max: \\u00a3{max_spread:,.0f}/MWh')"
    ))

    # ── Section 3: BM redispatch by carrier ───────────────────────────────────
    cells.append(_md("## 3. BM Redispatch — Volume by Carrier"))

    cells.append(_code(
        "fig, ax = plt.subplots(figsize=(12, 6))\n"
        "fig.suptitle(f'Balancing Mechanism — Redispatch Volume by Carrier ({SCENARIO})', fontsize=13)\n"
        "\n"
        "if 'carrier' in rd.columns:\n"
        "    by_c = rd.groupby('carrier')[['increase_MWh', 'decrease_MWh']].sum()\n"
        "    by_c = by_c[by_c.sum(axis=1) > 1].copy()\n"
        "    by_c['net'] = by_c['increase_MWh'] - by_c['decrease_MWh']\n"
        "    by_c = by_c.sort_values('net', ascending=False)\n"
        "\n"
        "    x = np.arange(len(by_c))\n"
        "    w = 0.35\n"
        "    ax.bar(x - w/2, by_c['increase_MWh'] / 1000, w, label='Increase (\\u2191)',\n"
        "           color='#2166ac', alpha=0.85)\n"
        "    ax.bar(x + w/2, -by_c['decrease_MWh'] / 1000, w, label='Decrease (\\u2193)',\n"
        "           color='#d6604d', alpha=0.85)\n"
        "    ax.set_xticks(x)\n"
        "    ax.set_xticklabels(by_c.index, rotation=45, ha='right', fontsize=8)\n"
        "    ax.axhline(0, color='black', linewidth=0.8)\n"
        "    ax.set_ylabel('Volume (GWh)')\n"
        "    ax.legend()\n"
        "else:\n"
        "    ax.text(0.5, 0.5, 'carrier column not found in redispatch CSV',\n"
        "            ha='center', va='center', transform=ax.transAxes)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Section 4: BM constraint costs ───────────────────────────────────────
    cells.append(_md("## 4. BM Constraint Costs by Carrier"))

    cells.append(_code(
        "carrier_col = cc.columns[0]\n"
        "cc_plot = cc[cc[carrier_col] != 'TOTAL'].copy()\n"
        "cc_plot = cc_plot[cc_plot['net_cost'].abs() > 1].copy()\n"
        "cc_plot = cc_plot.sort_values('net_cost', ascending=False).reset_index(drop=True)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n"
        "fig.suptitle(f'BM Constraint Costs — {SCENARIO}', fontsize=13)\n"
        "\n"
        "# Bar: offer vs bid cost\n"
        "ax = axes[0]\n"
        "x = np.arange(len(cc_plot))\n"
        "ax.bar(x, cc_plot['offer_cost'] / 1000, label='Offer cost (\\u2191)', color='#2166ac', alpha=0.85)\n"
        "ax.bar(x, cc_plot['bid_cost'] / 1000,   label='Bid cost (\\u2193)',   color='#d6604d', alpha=0.85,\n"
        "       bottom=cc_plot['offer_cost'] / 1000)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(cc_plot[carrier_col], rotation=45, ha='right', fontsize=8)\n"
        "ax.set_ylabel('Cost (\\u00a3k)')\n"
        "ax.set_title('Stacked offer/bid costs')\n"
        "ax.legend()\n"
        "total_cost = cc_plot['net_cost'].sum()\n"
        "ax.text(0.98, 0.95, f'Total BM cost: {fmt_gbp(total_cost)}',\n"
        "        transform=ax.transAxes, ha='right', va='top', fontsize=9,\n"
        "        bbox=dict(boxstyle='round', fc='lightyellow', ec='orange'))\n"
        "\n"
        "# Pie: cost breakdown\n"
        "ax2 = axes[1]\n"
        "threshold = total_cost * 0.01\n"
        "main = cc_plot[cc_plot['net_cost'] >= threshold].copy()\n"
        "other_val = cc_plot[cc_plot['net_cost'] < threshold]['net_cost'].sum()\n"
        "if other_val > 1:\n"
        "    main = pd.concat([main, pd.DataFrame([{carrier_col: 'Other', 'net_cost': other_val}])],\n"
        "                     ignore_index=True)\n"
        "ax2.pie(main['net_cost'], labels=main[carrier_col],\n"
        "        colors=[COLORS.get(c, '#999') for c in main[carrier_col]],\n"
        "        autopct='%1.1f%%', startangle=90,\n"
        "        explode=[0.05 if c == 'load_shedding' else 0 for c in main[carrier_col]],\n"
        "        textprops={'fontsize': 8})\n"
        "ax2.set_title(f'Cost breakdown\\nTotal: {fmt_gbp(total_cost)}')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Section 5: Top BM assets ──────────────────────────────────────────────
    cells.append(_md("## 5. Top BM Assets"))

    cells.append(_code(
        "comp_col = rd.columns[0]\n"
        "\n"
        "print(f'=== {SCENARIO} — Top 15 by Increase ===')\n"
        "show = [c for c in [comp_col, 'carrier', 'increase_MWh', 'offer_cost'] if c in rd.columns]\n"
        "df_up = rd.nlargest(15, 'increase_MWh')[show].copy()\n"
        "df_up['increase_MWh'] = df_up['increase_MWh'].map('{:,.0f}'.format)\n"
        "df_up['offer_cost']   = df_up['offer_cost'].map('\\u00a3{:,.0f}'.format)\n"
        "display(df_up.reset_index(drop=True))\n"
        "\n"
        "print()\n"
        "print(f'=== {SCENARIO} — Top 15 by Decrease ===')\n"
        "show2 = [c for c in [comp_col, 'carrier', 'decrease_MWh', 'bid_cost'] if c in rd.columns]\n"
        "df_dn = rd.nlargest(15, 'decrease_MWh')[show2].copy()\n"
        "df_dn['decrease_MWh'] = df_dn['decrease_MWh'].map('{:,.0f}'.format)\n"
        "df_dn['bid_cost']     = df_dn['bid_cost'].map('\\u00a3{:,.0f}'.format)\n"
        "display(df_dn.reset_index(drop=True))"
    ))

    # ── Section 6: Congestion ─────────────────────────────────────────────────
    cells.append(_md("## 6. Network Congestion"))

    cells.append(_code(
        "print(f'{SCENARIO}: {len(cg)} congested components')\n"
        "\n"
        "if len(cg) == 0:\n"
        "    print('  No congested components — BM resolve was uncongested.')\n"
        "else:\n"
        "    sort_col = 'hours_congested' if 'hours_congested' in cg.columns else cg.columns[0]\n"
        "    top_cg = cg.sort_values(sort_col, ascending=False).head(25).reset_index(drop=True)\n"
        "    comp_label = top_cg.columns[0]\n"
        "    y_labels = top_cg[comp_label].astype(str)\n"
        "\n"
        "    n_axes = 2 if 'mean_loading_fraction' in top_cg.columns else 1\n"
        "    fig, axes = plt.subplots(1, n_axes, figsize=(13, max(4, len(top_cg) * 0.25)))\n"
        "    if n_axes == 1:\n"
        "        axes = [axes]\n"
        "    fig.suptitle(f'BM Congestion — {SCENARIO}', fontsize=12)\n"
        "\n"
        "    axes[0].barh(range(len(top_cg)), top_cg['hours_congested'], color='#d62728', alpha=0.8)\n"
        "    axes[0].set_yticks(range(len(top_cg)))\n"
        "    axes[0].set_yticklabels(y_labels, fontsize=7)\n"
        "    axes[0].set_xlabel('Hours congested')\n"
        "    axes[0].set_title('Hours at Congestion Limit')\n"
        "    axes[0].invert_yaxis()\n"
        "\n"
        "    if n_axes == 2:\n"
        "        axes[1].barh(range(len(top_cg)), top_cg['mean_loading_fraction'] * 100,\n"
        "                     color='#e6550d', alpha=0.8)\n"
        "        axes[1].axvline(100, color='black', linewidth=1, linestyle='--')\n"
        "        axes[1].set_yticks(range(len(top_cg)))\n"
        "        axes[1].set_yticklabels(y_labels, fontsize=7)\n"
        "        axes[1].set_xlabel('Mean loading fraction (%)')\n"
        "        axes[1].set_title('Mean Loading Fraction')\n"
        "        axes[1].invert_yaxis()\n"
        "\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ))

    # ── Section 7: Price comparison ───────────────────────────────────────────
    cells.append(_md("## 7. Wholesale vs BM Nodal Prices"))

    cells.append(_code(
        "fig, axes = plt.subplots(2, 1, figsize=(12, 8))\n"
        "fig.suptitle(f'Wholesale vs BM Nodal Prices — {SCENARIO}', fontsize=13)\n"
        "\n"
        "ax = axes[0]\n"
        "if 'wholesale_price' in pc.columns:\n"
        "    ax.plot(pc.index, pc['wholesale_price'], 'b-o', ms=4, label='Wholesale price', zorder=3)\n"
        "if 'mean_nodal_price' in pc.columns:\n"
        "    ax.plot(pc.index, pc['mean_nodal_price'], 'r-s', ms=4, label='BM mean nodal', zorder=3)\n"
        "if 'min_nodal_price' in pc.columns and 'max_nodal_price' in pc.columns:\n"
        "    ax.fill_between(pc.index, pc['min_nodal_price'], pc['max_nodal_price'],\n"
        "                    alpha=0.15, color='red', label='BM nodal range')\n"
        "ax.axhline(0, color='black', linewidth=0.8, linestyle='--')\n"
        "ax.set_title('Price Time Series')\n"
        "ax.set_ylabel('\\u00a3/MWh')\n"
        "ax.legend(fontsize=9)\n"
        "ax.tick_params(axis='x', labelrotation=30)\n"
        "\n"
        "ax2 = axes[1]\n"
        "if 'nodal_spread' in pc.columns:\n"
        "    ax2.bar(range(len(pc)), pc['nodal_spread'], color='#e6550d', alpha=0.8)\n"
        "    mean_spread = pc['nodal_spread'].mean()\n"
        "    ax2.axhline(mean_spread, color='black', linestyle='--', linewidth=1)\n"
        "    ax2.set_title('BM Nodal Price Spread')\n"
        "    ax2.set_ylabel('\\u00a3/MWh spread')\n"
        "    ax2.set_xticks(range(len(pc)))\n"
        "    ax2.set_xticklabels([str(t)[:13] for t in pc.index], rotation=45, ha='right', fontsize=7)\n"
        "    ax2.text(0.98, 0.95, f'Mean spread: \\u00a3{mean_spread:.1f}/MWh',\n"
        "             transform=ax2.transAxes, ha='right', va='top', fontsize=9)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Section 8: Summary table ──────────────────────────────────────────────
    cells.append(_md("## 8. Summary Statistics"))

    cells.append(_code(
        "total_bm_arr = cc[cc[cc.columns[0]] == 'TOTAL']['net_cost'].values\n"
        "total_bm = total_bm_arr[0] if len(total_bm_arr) else rd['net_cost'].sum()\n"
        "\n"
        "ls_arr = cc[cc[cc.columns[0]] == 'load_shedding']['net_cost'].values\n"
        "ls_cost = ls_arr[0] if len(ls_arr) else 0\n"
        "\n"
        "summary = pd.DataFrame([{\n"
        "    'Scenario':                         SCENARIO,\n"
        "    'Timesteps':                        len(wp),\n"
        "    'Wholesale assets':                 len(gen.columns),\n"
        "    'BM assets total':                  len(rd),\n"
        "    'BM assets increased':              int((rd['increase_MWh'] > 0.1).sum()),\n"
        "    'BM assets decreased':              int((rd['decrease_MWh'] > 0.1).sum()),\n"
        "    'Total generation (GWh)':           f\"{gen.sum().sum()/1000:.1f}\",\n"
        "    'Wholesale price mean (\\u00a3/MWh)':    f\"{wp['wholesale_price'].mean():.1f}\",\n"
        "    'Wholesale price min (\\u00a3/MWh)':     f\"{wp['wholesale_price'].min():.1f}\",\n"
        "    'Wholesale price max (\\u00a3/MWh)':     f\"{wp['wholesale_price'].max():.1f}\",\n"
        "    'BM increase (GWh)':                f\"{rd['increase_MWh'].sum()/1000:.2f}\",\n"
        "    'BM decrease (GWh)':                f\"{rd['decrease_MWh'].sum()/1000:.2f}\",\n"
        "    'Total BM cost':                    fmt_gbp(total_bm),\n"
        "    'Load shedding cost':               fmt_gbp(ls_cost),\n"
        "    'Congested components':             len(cg),\n"
        "    'BM mean nodal spread (\\u00a3/MWh)':   (\n"
        "        f\"{pc['nodal_spread'].mean():.1f}\" if 'nodal_spread' in pc.columns else 'N/A'\n"
        "    ),\n"
        "}]).set_index('Scenario').T\n"
        "\n"
        "display(summary)"
    ))

    # ── Notebook metadata ─────────────────────────────────────────────────────
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    try:
        scenario = snakemake.wildcards.scenario
        inputs = {
            "wholesale_dispatch_csv":  snakemake.input.wholesale_dispatch_csv,
            "wholesale_storage_csv":   snakemake.input.wholesale_storage_csv,
            "wholesale_links_csv":     snakemake.input.wholesale_links_csv,
            "wholesale_price_csv":     snakemake.input.wholesale_price_csv,
            "balancing_dispatch_csv":  snakemake.input.balancing_dispatch_csv,
            "redispatch_summary_csv":  snakemake.input.redispatch_summary_csv,
            "constraint_costs_csv":    snakemake.input.constraint_costs_csv,
            "congestion_csv":          snakemake.input.congestion_csv,
            "price_comparison_csv":    snakemake.input.price_comparison_csv,
            "wholesale_network":       snakemake.input.wholesale_network,
            "balancing_network":       snakemake.input.balancing_network,
        }
        output_path = snakemake.output.notebook
    except NameError:
        # Standalone / testing usage
        import sys
        if len(sys.argv) < 3:
            print("Usage: python generate_market_analysis_notebook.py <scenario> <output.ipynb>")
            print("  Paths for market CSVs are inferred from resources/market/<scenario>_*.csv")
            sys.exit(1)
        scenario = sys.argv[1]
        base = f"resources/market/{scenario}"
        inputs = {
            "wholesale_dispatch_csv":  f"{base}_wholesale_dispatch.csv",
            "wholesale_storage_csv":   f"{base}_wholesale_storage.csv",
            "wholesale_links_csv":     f"{base}_wholesale_links.csv",
            "wholesale_price_csv":     f"{base}_wholesale_price.csv",
            "balancing_dispatch_csv":  f"{base}_balancing_dispatch.csv",
            "redispatch_summary_csv":  f"{base}_redispatch_summary.csv",
            "constraint_costs_csv":    f"{base}_constraint_costs.csv",
            "congestion_csv":          f"{base}_congestion.csv",
            "price_comparison_csv":    f"{base}_price_comparison.csv",
            "wholesale_network":       f"resources/market/{scenario}_wholesale.nc",
            "balancing_network":       f"resources/market/{scenario}_balancing.nc",
        }
        output_path = sys.argv[2]

    logger.info(f"Building market analysis notebook for scenario: {scenario}")

    nb = build_notebook(scenario, inputs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)

    logger.info(f"Notebook written to {output_path}")


main()
