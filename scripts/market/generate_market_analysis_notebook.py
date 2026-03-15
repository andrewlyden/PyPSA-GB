"""
Generate a per-scenario Jupyter notebook for two-stage market dispatch analysis.

This script is called by the Snakemake rule 'generate_market_analysis_notebook'.
It programmatically builds a self-contained .ipynb that mirrors the interactive
analysis in notebooks/two_stage_market_dispatch.ipynb, but parameterised for a
single scenario so it can be executed as part of the automated workflow.

The generated notebook structure matches the hand-crafted reference notebook:
  - Background section: GB market structure, architecture diagram, bid/offer prices
  - Setup: libraries and file paths
  - Load Results: all CSV files with explanatory table
  - Stage 1 — Wholesale Dispatch: generation + storage by carrier
  - Stage 1 — Wholesale Clearing Price: SMP time series + copperplate diagnostic
  - Stage 2 — BM Redispatch Volumes: by carrier
  - Stage 2 — BM Constraint Costs: stacked bar + pie
  - Stage 2 — Top BM Assets: top 15 by increase / decrease
  - Stage 2 — Network Congestion: hours + mean loading fraction
  - Wholesale vs BM Nodal Prices: time series + spread
  - Summary Statistics: full metric table
  - Further Reading: config example, key source files, known limitations

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

    # ── Title & background ────────────────────────────────────────────────────
    cells.append(_md(
        f"# Two-Stage Market Dispatch in PyPSA-GB — `{scenario}`\n"
        "\n"
        "This notebook walks through the two-stage wholesale + balancing mechanism (BM) "
        "market dispatch results for the scenario above.\n"
        "\n"
        "---\n"
        "\n"
        "## Background: How GB Electricity Markets Work\n"
        "\n"
        "The GB electricity market operates in two distinct stages:\n"
        "\n"
        "| Stage | Market | Boundary | What happens |\n"
        "|-------|--------|----------|--------------|\n"
        "| **1** | **Wholesale / Day-Ahead** | Gate closure (~1 hr ahead) | "
        "Generators submit bids/offers; a uniform clearing price is set assuming no "
        "network constraints (\"copperplate\") |\n"
        "| **2** | **Balancing Mechanism (BM)** | Real-time | National Grid ESO (NESO) "
        "resolves network constraints by accepting bids (decrease) and offers (increase) "
        "from generators |\n"
        "\n"
        "The key feature is the **separation of economics from physics**: the wholesale "
        "market sets the price, and the BM corrects for the physical reality of network "
        "congestion. Generators are paid **twice**:\n"
        "- Their **wholesale revenue** (energy × wholesale price)\n"
        "- **BM payments** for accepting NESO's instructions to deviate from their "
        "wholesale position\n"
        "\n"
        "---\n"
        "\n"
        "## PyPSA-GB Model Architecture\n"
        "\n"
        "```\n"
        "Network (.nc)\n"
        "    │\n"
        "    ├─── Stage 1: solve_wholesale.py\n"
        "    │       • Relax all line/transformer s_nom → 1,000,000 MW  (copperplate)\n"
        "    │       • Solve LP: minimise Σ marginal_cost · dispatch\n"
        "    │       • Extract: wholesale dispatch, uniform price\n"
        "    │\n"
        "    └─── Stage 2: solve_balancing.py\n"
        "            • Restore original s_nom (full network constraints)\n"
        "            • Fix generators to wholesale positions via linopy constraints:\n"
        "                  p = p_wholesale + increase − decrease\n"
        "            • Objective: minimise Σ offer·increase + bid·decrease\n"
        "            • Extract: BM redispatch volumes, nodal prices, congested lines\n"
        "```\n"
        "\n"
        "---\n"
        "\n"
        "## Bid and Offer Prices\n"
        "\n"
        "Because generators already have a wholesale contract, their BM prices represent "
        "the **marginal value of deviating**:\n"
        "\n"
        "$$\\text{offer price} = MC \\times (1 + \\text{offer markup})$$\n"
        "$$\\text{bid price} = \\max\\bigl(MC \\times (1 - \\text{bid discount}),\\ 0.50\\bigr)$$\n"
        "\n"
        "Default markups (configurable per carrier in `defaults.yaml`):\n"
        "\n"
        "| Carrier | Offer markup | Bid discount |\n"
        "|---------|-------------|-------------|\n"
        "| Default | +10% | −10% |\n"
        "| Nuclear | +50% | −5% |\n"
        "| Wind (onshore/offshore) | 0% | −5% |\n"
        "| Battery | +15% | −15% |\n"
        "\n"
        "*Generated automatically by the PyPSA-GB Snakemake workflow.*"
    ))

    # ── Setup ─────────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Setup — Libraries and File Paths\n"
        "\n"
        "The following cell loads standard libraries and resolves file paths to the "
        f"`{scenario}` results under `resources/market/`. The path discovery logic "
        "walks up from the working directory to find the repository root, so this "
        "notebook runs correctly from any working directory."
    ))

    rel_paths = {}
    for key, path_val in inputs.items():
        rel = os.path.relpath(path_val, start=Path.cwd())
        rel_paths[key] = rel.replace("\\", "/")

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
        "# ── Scenario ─────────────────────────────────────────────────────────\n"
        f"SCENARIO = {repr(scenario)}\n"
        "\n"
        "# ── File paths (relative from repo root) ────────────────────────────\n"
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
        "# ── Resolve absolute paths ────────────────────────────────────────────\n"
        "cwd = Path(os.getcwd())\n"
        "def _find_base(path):\n"
        "    if (path / 'resources' / 'market').exists():\n"
        "        return path\n"
        "    for parent in path.parents:\n"
        "        if (parent / 'resources' / 'market').exists():\n"
        "            return parent\n"
        "    return path\n"
        "base = _find_base(cwd)\n"
        "for k, v in list(PATHS.items()):\n"
        "    p = Path(v)\n"
        "    if not p.is_absolute():\n"
        "        candidate = base / v\n"
        "        PATHS[k] = str(candidate.resolve() if candidate.exists() else (cwd / v).resolve())\n"
        "\n"
        "# ── Helper formatters ─────────────────────────────────────────────────\n"
        "COLORS = {\n"
        "    'wind_offshore': '#1f77b4', 'wind_onshore': '#aec7e8', 'solar_pv': '#ffdd57',\n"
        "    'nuclear': '#9467bd', 'CCGT': '#e86414', 'Battery': '#2ca02c',\n"
        "    'Pumped Storage Hydroelectricity': '#17becf', 'large_hydro': '#98df8a',\n"
        "    'load_shedding': '#d62728', 'OCGT': '#fd8d3c', 'waste_to_energy': '#8c564b',\n"
        "    'landfill_gas': '#bcbd22', 'biogas': '#6baed6', 'marine': '#3182bd',\n"
        "    'LAES': '#74c476', 'Domestic Battery': '#7fc97f',\n"
        "}\n"
        "\n"
        "def fmt_gbp(x, pos=None):\n"
        "    if abs(x) >= 1e6: return f'\\u00a3{x/1e6:.1f}M'\n"
        "    if abs(x) >= 1e3: return f'\\u00a3{x/1e3:.0f}k'\n"
        "    return f'\\u00a3{x:.0f}'\n"
        "\n"
        f"print(f'Scenario : {scenario}')\n"
        "print(f'Repo root: {base}')\n"
        "print('Setup complete.')"
    ))

    # ── Load results ──────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Load Results\n"
        "\n"
        "All outputs from the two-stage solve are stored as CSVs under `resources/market/`. "
        "The cell below loads them all into memory. The key files are:\n"
        "\n"
        "| File | Contents |\n"
        "|------|----------|\n"
        "| `*_wholesale_dispatch.csv` | Generator dispatch (MW) at each timestep — Stage 1 |\n"
        "| `*_wholesale_price.csv` | Uniform wholesale clearing price (£/MWh) + bus spread |\n"
        "| `*_balancing_dispatch.csv` | Generator dispatch (MW) after BM redispatch — Stage 2 |\n"
        "| `*_redispatch_summary.csv` | Per-asset increase/decrease volumes and BM costs |\n"
        "| `*_constraint_costs.csv` | BM costs aggregated by carrier |\n"
        "| `*_congestion.csv` | Congested lines/transformers and loading fractions |\n"
        "| `*_price_comparison.csv` | Wholesale vs BM nodal price comparison |\n"
        "| `*_wholesale.nc` / `*_balancing.nc` | Full PyPSA network files (for detailed analysis) |"
    ))

    cells.append(_code(
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
        "print(f'  Timesteps            : {len(wp)}')\n"
        "print(f'  Generator assets (WS): {len(gen.columns)}')\n"
        "print(f'  Storage assets (WS)  : {len(su.columns)}')\n"
        "print(f'  Link assets (WS)     : {len(lnk.columns)}')\n"
        "print(f'  BM redispatch assets : {len(rd)}')\n"
        "print(f'  BM congested comps   : {len(cg)}')"
    ))

    # ── Stage 1: Wholesale dispatch ───────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Stage 1: Wholesale Dispatch\n"
        "\n"
        "### What happens in Stage 1?\n"
        "\n"
        "The wholesale solve relaxes all AC line and transformer thermal limits to "
        "1,000,000 MW — effectively making the entire GB network one giant \"copperplate\" "
        "node. The LP then minimises total generation cost:\n"
        "\n"
        "$$\\min \\sum_{g,t} MC_g \\cdot p_{g,t}$$\n"
        "\n"
        "subject only to power balance constraints (supply = demand at every timestep). "
        "Because there are no transmission constraints, every bus has the **same marginal "
        "price** — this is the **System Marginal Price (SMP)** or wholesale clearing price.\n"
        "\n"
        "The merit order determines which generators run:\n"
        "1. Zero-cost renewables (wind, solar, hydro) run first\n"
        "2. Nuclear (low marginal cost but often constrained by must-run or minimum output)\n"
        "3. Gas (CCGT) fills the remaining gap\n"
        "4. Open-cycle gas (OCGT) or load shedding (£6,000/MWh) only if supply is tight\n"
        "\n"
        "The chart below shows total generation by carrier across all timesteps in the "
        "solve period."
    ))

    cells.append(_code(
        "import pypsa\n"
        "\n"
        "def dispatch_by_carrier(dispatch_df, network_path):\n"
        "    \"\"\"Sum generator dispatch (MWh) by carrier using the network for carrier lookup.\"\"\"\n"
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
        "    \"\"\"Sum storage dispatch (MWh) by carrier.\"\"\"\n"
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
        f"fig.suptitle(f'Stage 1 \\u2014 Wholesale Dispatch by Carrier ({scenario})', fontsize=13)\n"
        "\n"
        "ax = axes[0]\n"
        "if isinstance(gen_by_c, pd.Series):\n"
        "    plot = gen_by_c[(gen_by_c.abs() > 100) & (gen_by_c.index != 'load_shedding')]\n"
        "    plot = plot.sort_values(ascending=False)\n"
        "    plot.div(1000).plot.bar(ax=ax,\n"
        "        color=[COLORS.get(c, '#999') for c in plot.index],\n"
        "        edgecolor='white', linewidth=0.5)\n"
        "    ax.set_title('Generation by Carrier (generators only)')\n"
        "    ax.set_ylabel('Energy (GWh)')\n"
        "    ax.tick_params(axis='x', labelrotation=45)\n"
        "    ax.xaxis.set_tick_params(labelsize=8)\n"
        "\n"
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

    # ── Stage 1: Wholesale price ──────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Stage 1: Wholesale Clearing Price\n"
        "\n"
        "### Interpreting the price\n"
        "\n"
        "In a copperplate solve, **every demand bus has the same marginal price** "
        "(the dual variable of the nodal power balance constraint). The price equals "
        "the marginal cost of the most expensive generator dispatched — i.e., the "
        "generator on the margin.\n"
        "\n"
        "The **price spread** across buses is a diagnostic: in a true copperplate solve "
        "it should be essentially zero (≤ £0.01/MWh). A non-zero spread indicates "
        "numerical solver tolerance effects — normal for large networks.\n"
        "\n"
        "> **Note on bus filtering:** The spread is computed over GB demand buses only. "
        "DC link buses (H2 turbines, electrolysers, internal HVDC) carry directional "
        "cost differentials and are excluded to avoid inflating the reported spread."
    ))

    cells.append(_code(
        "fig, axes = plt.subplots(2, 1, figsize=(12, 7))\n"
        f"fig.suptitle(f'Stage 1 \\u2014 Wholesale Clearing Price ({scenario})', fontsize=13)\n"
        "\n"
        "ax = axes[0]\n"
        "ax.plot(wp.index, wp['wholesale_price'], marker='o', color='#1f77b4', label='Wholesale price')\n"
        "ax.axhline(0, color='black', linewidth=0.8, linestyle='--')\n"
        "ax.set_title('System Marginal Price (SMP) \\u2014 marginal cost of the dispatched generator on the margin')\n"
        "ax.set_ylabel('\\u00a3/MWh')\n"
        "ax.legend(fontsize=9)\n"
        "ax.tick_params(axis='x', labelrotation=30)\n"
        "\n"
        "ax2 = axes[1]\n"
        "max_spread = wp['price_spread'].max()\n"
        "spread_color = '#d62728' if max_spread > 10 else '#2ca02c'\n"
        "ax2.plot(wp.index, wp['price_spread'], marker='s', color=spread_color)\n"
        "ax2.set_title(\n"
        "    'Price Spread Across Buses \\u2014 copperplate diagnostic (should be ~0)\\n'\n"
        "    'Non-zero values = numerical solver tolerance; large values = non-GB bus contamination'\n"
        ")\n"
        "ax2.set_ylabel('\\u00a3/MWh spread')\n"
        "ax2.tick_params(axis='x', labelrotation=30)\n"
        "ax2.text(0.02, 0.95, f'Max spread: \\u00a3{max_spread:,.4f}/MWh',\n"
        "         transform=ax2.transAxes, va='top', color=spread_color, fontsize=9)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print(f'Wholesale price  mean : \\u00a3{wp[\"wholesale_price\"].mean():.2f}/MWh')\n"
        "print(f'Wholesale price  min  : \\u00a3{wp[\"wholesale_price\"].min():.2f}/MWh')\n"
        "print(f'Wholesale price  max  : \\u00a3{wp[\"wholesale_price\"].max():.2f}/MWh')\n"
        "print(f'Copperplate spread max: \\u00a3{max_spread:,.4f}/MWh')"
    ))

    # ── Stage 2: BM redispatch volumes ────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Stage 2: Balancing Mechanism — Redispatch Volumes\n"
        "\n"
        "### What happens in Stage 2?\n"
        "\n"
        "The BM solve restores full network physics (real `s_nom` on every line and "
        "transformer) and anchors each generator to its wholesale position using "
        "additional linopy constraints:\n"
        "\n"
        "$$p_{g,t} = p^{\\text{wholesale}}_{g,t} + \\text{increase}_{g,t} - \\text{decrease}_{g,t}$$\n"
        "\n"
        "The new minimum-cost objective is:\n"
        "\n"
        "$$\\min \\sum_{g,t} \\left( \\text{offer}_g \\cdot \\text{increase}_{g,t} + "
        "\\text{bid}_g \\cdot \\text{decrease}_{g,t} \\right)$$\n"
        "\n"
        "The BM clears congestion by paying generators *north of a constraint* to reduce "
        "output (accept their bid) and generators *south of a constraint* to increase "
        "output (accept their offer).\n"
        "\n"
        "### What do the redispatch volumes tell us?\n"
        "\n"
        "- **Large increases in a region** → that region is import-constrained; NESO pays "
        "local generators extra to run up\n"
        "- **Large decreases in a region** → that region is export-constrained; NESO pays "
        "generators to reduce\n"
        "- **Symmetric increase = decrease** → the BM is balancing a constraint between "
        "two regions"
    ))

    cells.append(_code(
        "fig, ax = plt.subplots(figsize=(12, 6))\n"
        f"fig.suptitle(f'Stage 2 \\u2014 Balancing Mechanism Redispatch Volume by Carrier ({scenario})', fontsize=13)\n"
        "\n"
        "if 'carrier' in rd.columns:\n"
        "    by_c = rd.groupby('carrier')[['increase_MWh', 'decrease_MWh']].sum()\n"
        "    by_c = by_c[by_c.sum(axis=1) > 1].copy()\n"
        "    by_c['net'] = by_c['increase_MWh'] - by_c['decrease_MWh']\n"
        "    by_c = by_c.sort_values('net', ascending=False)\n"
        "\n"
        "    x = np.arange(len(by_c))\n"
        "    w = 0.35\n"
        "    ax.bar(x - w/2, by_c['increase_MWh'] / 1000, w, label='Increase (\\u2191) \\u2014 offer accepted',\n"
        "           color='#2166ac', alpha=0.85)\n"
        "    ax.bar(x + w/2, -by_c['decrease_MWh'] / 1000, w, label='Decrease (\\u2193) \\u2014 bid accepted',\n"
        "           color='#d6604d', alpha=0.85)\n"
        "    ax.set_xticks(x)\n"
        "    ax.set_xticklabels(by_c.index, rotation=45, ha='right', fontsize=8)\n"
        "    ax.axhline(0, color='black', linewidth=0.8)\n"
        "    ax.set_ylabel('Volume (GWh)')\n"
        "    ax.set_xlabel('Carrier')\n"
        "    ax.legend()\n"
        "    ax.set_title(\n"
        "        'Positive bars = NESO accepted offers (generators ran up)\\n'\n"
        "        'Negative bars = NESO accepted bids (generators ran down)'\n"
        "    )\n"
        "else:\n"
        "    ax.text(0.5, 0.5, 'carrier column not found in redispatch CSV',\n"
        "            ha='center', va='center', transform=ax.transAxes)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Stage 2: Constraint costs ─────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Stage 2: BM Constraint Costs by Carrier\n"
        "\n"
        "### BM cost structure\n"
        "\n"
        "The BM cost is the **system cost of resolving network constraints**. It is paid "
        "by consumers via the Balancing Services Use of System (BSUoS) charge in the "
        "real market.\n"
        "\n"
        "$$\\text{BM cost} = \\sum_g \\bigl( \\text{offer}_g \\cdot \\text{increase}_g + "
        "\\text{bid}_g \\cdot \\text{decrease}_g \\bigr)$$\n"
        "\n"
        "Key observations to look for:\n"
        "- **Load shedding** appearing in BM costs signals a genuine capacity shortage "
        "*after* network constraints are imposed (a bus is isolated or constrained off)\n"
        "- **High CCGT costs** typically mean gas plant near congested interconnections "
        "is being re-dispatched to manage flows\n"
        "- **High renewables decrease** costs suggest excess renewable generation "
        "constrained behind a bottleneck"
    ))

    cells.append(_code(
        "carrier_col = cc.columns[0]\n"
        "cc_plot = cc[cc[carrier_col] != 'TOTAL'].copy()\n"
        "cc_plot = cc_plot[cc_plot['net_cost'].abs() > 1].copy()\n"
        "cc_plot = cc_plot.sort_values('net_cost', ascending=False).reset_index(drop=True)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n"
        f"fig.suptitle(f'Stage 2 \\u2014 BM Constraint Costs by Carrier ({scenario})', fontsize=13)\n"
        "\n"
        "ax = axes[0]\n"
        "x = np.arange(len(cc_plot))\n"
        "ax.bar(x, cc_plot['offer_cost'] / 1000, label='Offer cost (\\u2191 accepted)', color='#2166ac', alpha=0.85)\n"
        "ax.bar(x, cc_plot['bid_cost'] / 1000, label='Bid cost (\\u2193 accepted)', color='#d6604d', alpha=0.85,\n"
        "       bottom=cc_plot['offer_cost'] / 1000)\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(cc_plot[carrier_col], rotation=45, ha='right', fontsize=8)\n"
        "ax.set_ylabel('Cost (\\u00a3k)')\n"
        "ax.set_title('Stacked offer/bid costs by carrier')\n"
        "ax.legend()\n"
        "total_cost = cc_plot['net_cost'].sum()\n"
        "ax.text(0.98, 0.95, f'Total BM cost: {fmt_gbp(total_cost)}',\n"
        "        transform=ax.transAxes, ha='right', va='top', fontsize=9,\n"
        "        bbox=dict(boxstyle='round', fc='lightyellow', ec='orange'))\n"
        "\n"
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
        "ax2.set_title(f'BM cost share by carrier\\nTotal: {fmt_gbp(total_cost)}')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Stage 2: Top BM assets ────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Stage 2: Top BM Assets\n"
        "\n"
        "This table identifies the individual generators with the largest BM activity — "
        "both those **increased** (running up to relieve import constraints) and those "
        "**decreased** (running down to relieve export constraints). In the real BM, "
        "these would be the generators receiving the most BM instruction notices (BOAs/SOs)."
    ))

    cells.append(_code(
        "comp_col = rd.columns[0]\n"
        "\n"
        "print(f'=== {SCENARIO} \\u2014 Top 15 by Increase (offer accepted) ===')\n"
        "show = [c for c in [comp_col, 'carrier', 'increase_MWh', 'offer_cost'] if c in rd.columns]\n"
        "df_up = rd.nlargest(15, 'increase_MWh')[show].copy()\n"
        "df_up['increase_MWh'] = df_up['increase_MWh'].map('{:,.0f}'.format)\n"
        "df_up['offer_cost']   = df_up['offer_cost'].map('\\u00a3{:,.0f}'.format)\n"
        "display(df_up.reset_index(drop=True))\n"
        "\n"
        "print()\n"
        "print(f'=== {SCENARIO} \\u2014 Top 15 by Decrease (bid accepted) ===')\n"
        "show2 = [c for c in [comp_col, 'carrier', 'decrease_MWh', 'bid_cost'] if c in rd.columns]\n"
        "df_dn = rd.nlargest(15, 'decrease_MWh')[show2].copy()\n"
        "df_dn['decrease_MWh'] = df_dn['decrease_MWh'].map('{:,.0f}'.format)\n"
        "df_dn['bid_cost']     = df_dn['bid_cost'].map('\\u00a3{:,.0f}'.format)\n"
        "display(df_dn.reset_index(drop=True))"
    ))

    # ── Stage 2: Congestion ───────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Stage 2: Network Congestion\n"
        "\n"
        "### What is congestion in this context?\n"
        "\n"
        "A line or transformer is **congested** when the BM solve tries to push more power "
        "through it than its thermal rating allows. The BM congestion analysis identifies "
        "components loaded above 95% of their `s_nom` rating.\n"
        "\n"
        "In the real network, persistent congestion at specific boundaries (e.g. the B6 "
        "Scotland–England boundary, or radial 132kV feeders) drives significant BM costs. "
        "The ETYS network model here has 2,000+ buses and 3,700+ branches, so congestion "
        "can occur at realistic constraint locations.\n"
        "\n"
        "> **Tip:** If there is no congestion after Stage 2, the wholesale dispatch was "
        "already physically feasible — the BM makes no constraint-driven changes and total "
        "BM cost ≈ 0."
    ))

    cells.append(_code(
        "print(f'{SCENARIO}: {len(cg)} congested components (>95% loading threshold)')\n"
        "\n"
        "if len(cg) == 0:\n"
        "    print('  No congestion \\u2014 the wholesale dispatch was already network-feasible.')\n"
        "    print('  BM cost arises only from bid/offer markup rather than constraint management.')\n"
        "else:\n"
        "    sort_col = 'hours_congested' if 'hours_congested' in cg.columns else cg.columns[0]\n"
        "    top_cg = cg.sort_values(sort_col, ascending=False).head(25).reset_index(drop=True)\n"
        "    comp_label = top_cg.columns[0]\n"
        "    y_labels = top_cg[comp_label].astype(str)\n"
        "\n"
        "    n_ax = 2 if 'mean_loading_fraction' in top_cg.columns else 1\n"
        "    fig, axes = plt.subplots(1, n_ax, figsize=(13, max(4, len(top_cg) * 0.28)))\n"
        "    if n_ax == 1:\n"
        "        axes = [axes]\n"
        f"    fig.suptitle(f'Stage 2 \\u2014 Network Congestion (top 25) \\u2014 {scenario}', fontsize=12)\n"
        "\n"
        "    axes[0].barh(range(len(top_cg)), top_cg['hours_congested'], color='#d62728', alpha=0.8)\n"
        "    axes[0].set_yticks(range(len(top_cg)))\n"
        "    axes[0].set_yticklabels(y_labels, fontsize=7)\n"
        "    axes[0].set_xlabel('Hours congested')\n"
        "    axes[0].set_title('Hours at >95% loading')\n"
        "    axes[0].invert_yaxis()\n"
        "\n"
        "    if n_ax == 2:\n"
        "        axes[1].barh(range(len(top_cg)), top_cg['mean_loading_fraction'] * 100,\n"
        "                     color='#e6550d', alpha=0.8)\n"
        "        axes[1].axvline(100, color='black', linewidth=1, linestyle='--')\n"
        "        axes[1].set_yticks(range(len(top_cg)))\n"
        "        axes[1].set_yticklabels(y_labels, fontsize=7)\n"
        "        axes[1].set_xlabel('Mean loading fraction (%)')\n"
        "        axes[1].set_title('Mean loading fraction')\n"
        "        axes[1].invert_yaxis()\n"
        "\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ))

    # ── Wholesale vs BM nodal prices ──────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Wholesale vs BM Nodal Prices\n"
        "\n"
        "### What the price comparison shows\n"
        "\n"
        "| Price | Definition |\n"
        "|-------|------------|\n"
        "| Wholesale price | Uniform SMP from Stage 1 copperplate solve (£/MWh, same everywhere) |\n"
        "| BM mean nodal price | Average of per-bus dual variables in Stage 2 (varies by location) |\n"
        "| BM nodal spread | Max − Min across GB demand buses in Stage 2 |\n"
        "\n"
        "The **nodal spread** in Stage 2 is the key measure of "
        "**locational marginal pricing (LMP)** — how much the network constraints are "
        "worth in monetary terms. A spread of, say, £30/MWh across Scotland and south "
        "England means:\n"
        "\n"
        "> Generators *south* of a bottleneck would be paid £30/MWh more than generators "
        "*north* if the GB operated a nodal pricing market (as in the US PJM or MISO markets). "
        "The BM approximates this through cash-out.\n"
        "\n"
        "GB does **not** operate nodal pricing — the wholesale price is uniform and the BM "
        "handles constraint costs as a post-settlement adjustment. This model simulates "
        "the equivalent economic outcome."
    ))

    cells.append(_code(
        "fig, axes = plt.subplots(2, 1, figsize=(12, 8))\n"
        f"fig.suptitle(f'Wholesale vs BM Nodal Prices \\u2014 {scenario}', fontsize=13)\n"
        "\n"
        "ax = axes[0]\n"
        "if 'wholesale_price' in pc.columns:\n"
        "    ax.plot(pc.index, pc['wholesale_price'], 'b-o', ms=4, label='Wholesale (Stage 1 SMP)', zorder=3)\n"
        "if 'mean_nodal_price' in pc.columns:\n"
        "    ax.plot(pc.index, pc['mean_nodal_price'], 'r-s', ms=4, label='BM mean nodal (Stage 2)', zorder=3)\n"
        "if 'min_nodal_price' in pc.columns and 'max_nodal_price' in pc.columns:\n"
        "    ax.fill_between(pc.index, pc['min_nodal_price'], pc['max_nodal_price'],\n"
        "                    alpha=0.15, color='red', label='BM nodal range (min\\u2013max)')\n"
        "ax.axhline(0, color='black', linewidth=0.8, linestyle='--')\n"
        "ax.set_title(\n"
        "    'Price time series: uniform wholesale vs spatially-varying BM nodal prices'\n"
        ")\n"
        "ax.set_ylabel('\\u00a3/MWh')\n"
        "ax.legend(fontsize=9)\n"
        "ax.tick_params(axis='x', labelrotation=30)\n"
        "\n"
        "ax2 = axes[1]\n"
        "if 'nodal_spread' in pc.columns:\n"
        "    ax2.bar(range(len(pc)), pc['nodal_spread'], color='#e6550d', alpha=0.8)\n"
        "    mean_spread = pc['nodal_spread'].mean()\n"
        "    ax2.axhline(mean_spread, color='black', linestyle='--', linewidth=1,\n"
        "                label=f'Mean: \\u00a3{mean_spread:.1f}/MWh')\n"
        "    ax2.set_title(\n"
        "        'BM Nodal Spread \\u2014 the \\\"LMP shadow price\\\" of network constraints\\n'\n"
        "        'Larger spread = more binding constraints = higher constraint cost'\n"
        "    )\n"
        "    ax2.set_ylabel('\\u00a3/MWh spread')\n"
        "    ax2.set_xticks(range(len(pc)))\n"
        "    ax2.set_xticklabels([str(t)[:13] for t in pc.index], rotation=45, ha='right', fontsize=7)\n"
        "    ax2.legend(fontsize=9)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Summary statistics ────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Summary Statistics\n"
        "\n"
        "The table below summarises the full two-stage dispatch result for the selected "
        "scenario and solve period."
    ))

    cells.append(_code(
        "total_bm_arr = cc[cc[cc.columns[0]] == 'TOTAL']['net_cost'].values\n"
        "total_bm = total_bm_arr[0] if len(total_bm_arr) else rd['net_cost'].sum()\n"
        "\n"
        "ls_arr = cc[cc[cc.columns[0]] == 'load_shedding']['net_cost'].values\n"
        "ls_cost = ls_arr[0] if len(ls_arr) else 0\n"
        "\n"
        "summary = pd.DataFrame([{\n"
        "    'Scenario':                             SCENARIO,\n"
        "    'Timesteps':                            len(wp),\n"
        "    'Stage 1 \\u2014 wholesale assets':      len(gen.columns),\n"
        "    'Stage 2 \\u2014 BM assets total':       len(rd),\n"
        "    'Stage 2 \\u2014 assets increased (\\u2191)': int((rd['increase_MWh'] > 0.1).sum()),\n"
        "    'Stage 2 \\u2014 assets decreased (\\u2193)': int((rd['decrease_MWh'] > 0.1).sum()),\n"
        "    'Total generation (GWh)':               f\"{gen.sum().sum()/1000:.1f}\",\n"
        "    'Wholesale price \\u2014 mean (\\u00a3/MWh)': f\"{wp['wholesale_price'].mean():.2f}\",\n"
        "    'Wholesale price \\u2014 min (\\u00a3/MWh)':  f\"{wp['wholesale_price'].min():.2f}\",\n"
        "    'Wholesale price \\u2014 max (\\u00a3/MWh)':  f\"{wp['wholesale_price'].max():.2f}\",\n"
        "    'BM increase volume (GWh)':             f\"{rd['increase_MWh'].sum()/1000:.2f}\",\n"
        "    'BM decrease volume (GWh)':             f\"{rd['decrease_MWh'].sum()/1000:.2f}\",\n"
        "    'Total BM constraint cost':             fmt_gbp(total_bm),\n"
        "    'Load shedding BM cost':                fmt_gbp(ls_cost),\n"
        "    'Congested components (>95%)':          len(cg),\n"
        "    'BM mean nodal spread (\\u00a3/MWh)':   (\n"
        "        f\"{pc['nodal_spread'].mean():.1f}\" if 'nodal_spread' in pc.columns else 'N/A'\n"
        "    ),\n"
        "}]).set_index('Scenario').T\n"
        "\n"
        "display(summary)"
    ))

    # ── Further reading ───────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "\n"
        "## Further Reading & Configuration\n"
        "\n"
        "### How to run the two-stage market dispatch\n"
        "\n"
        "Any scenario can be run with the market dispatch enabled by setting "
        "`market.enabled: true` in `config/scenarios.yaml` or `config/defaults.yaml`:\n"
        "\n"
        "```yaml\n"
        "My_Scenario:\n"
        "  modelled_year: 2030\n"
        "  network_model: ETYS\n"
        "  market:\n"
        "    enabled: true\n"
        "    wholesale:\n"
        "      transmission_relaxation: 1.0e6   # copperplate MVA rating\n"
        "    balancing:\n"
        "      bid_offer_source: derived\n"
        "      default_offer_markup: 0.10\n"
        "      default_bid_discount: 0.10\n"
        "      carrier_overrides:\n"
        "        nuclear:        {offer_markup: 0.50, bid_discount: 0.05}\n"
        "        wind_onshore:   {offer_markup: 0.00, bid_discount: 0.05}\n"
        "        battery:        {offer_markup: 0.15, bid_discount: 0.15}\n"
        "      fix_interconnectors: true\n"
        "```\n"
        "\n"
        "Then run:\n"
        "```bash\n"
        "conda activate pypsa-gb\n"
        "snakemake -j 4 --latency-wait 60\n"
        "```\n"
        "\n"
        "### Key source files\n"
        "\n"
        "| File | Role |\n"
        "|------|------|\n"
        "| `scripts/market/solve_wholesale.py` | Stage 1 implementation |\n"
        "| `scripts/market/solve_balancing.py` | Stage 2 implementation |\n"
        "| `scripts/market/market_utils.py` | Bid/offer pricing, redispatch computation, congestion analysis |\n"
        "| `scripts/market/analyze_market.py` | Dashboard and summary generation |\n"
        "| `rules/market.smk` | Snakemake rules wiring the stages together |\n"
        "| `config/defaults.yaml` | Default market configuration |\n"
        "\n"
        "### Known limitations\n"
        "\n"
        "1. **Bid/offer prices are derived** from marginal costs — real BM participants "
        "use strategic pricing that may differ significantly\n"
        "2. **Unit commitment** is not modelled in LP mode — start-up and no-load costs "
        "are excluded\n"
        "3. **Intra-day redispatch** is not sequential — the BM is solved as a single "
        "batch LP over all timesteps\n"
        "4. **Interconnector flows** in historical scenarios are fixed to ESPENI observed "
        "values — they do not re-optimise in Stage 2"
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
