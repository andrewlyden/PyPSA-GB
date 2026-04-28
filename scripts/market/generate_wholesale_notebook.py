"""
Generate a per-scenario Jupyter notebook for wholesale market dispatch analysis.

Notebook sections:
  1. Setup — libraries, file paths, data loading
  2. Merit Order — supply curve (unit-level)
  3. Dispatch Stack — stacked area chart by carrier
  4. Wholesale Price vs Market — SMP overlaid with ELEXON MID
"""

import json
import logging
import os
from pathlib import Path

# ── Logging ─────────────────────────────────────────────────────────────────
try:
    log_file = snakemake.log[0]
except (NameError, AttributeError, IndexError, TypeError):
    log_file = "generate_wholesale_notebook.log"

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

def build_notebook(scenario: str, inputs: dict, scenario_config: dict) -> dict:
    """Return a complete nbformat-4 notebook for the given scenario."""

    cells = []

    # Resolve relative paths for portability
    rel_paths = {}
    for key, path_val in inputs.items():
        rel = os.path.relpath(path_val, start=Path.cwd())
        rel_paths[key] = rel.replace("\\", "/")

    modelled_year = scenario_config.get("modelled_year", 2023)
    sp = scenario_config.get("solve_period", {})
    solve_start = sp.get("start", "")
    solve_end = sp.get("end", "")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Setup
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md(
        f"# Wholesale Market Analysis — `{scenario}`\n"
        "\n"
        "Copperplate dispatch (all AC line limits relaxed). "
        "The LP minimises total generation cost; the uniform price is the "
        "**System Marginal Price (SMP)** — the marginal cost of the last "
        "generator dispatched.\n"
        "\n"
        "This notebook validates the modelled SMP against the **ELEXON Market "
        "Index Data (MID)** real wholesale price."
    ))

    cells.append(_code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import pypsa\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.ticker as mticker\n"
        "from matplotlib.patches import Patch\n"
        "import warnings, os, sys\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        f"SCENARIO     = {repr(scenario)}\n"
        f"MODELLED_YEAR = {modelled_year}\n"
        f"SOLVE_START  = {repr(solve_start)}\n"
        f"SOLVE_END    = {repr(solve_end)}\n"
        "\n"
        "PATHS = {\n"
        f"    'wholesale_network':  {repr(rel_paths['wholesale_network'])},\n"
        f"    'wholesale_dispatch': {repr(rel_paths['wholesale_dispatch_csv'])},\n"
        f"    'wholesale_storage':  {repr(rel_paths['wholesale_storage_csv'])},\n"
        f"    'wholesale_links':    {repr(rel_paths['wholesale_links_csv'])},\n"
        f"    'wholesale_price':    {repr(rel_paths['wholesale_price_csv'])},\n"
        f"    'marginal_costs':     {repr(rel_paths['marginal_costs_csv'])},\n"
        "}\n"
        "\n"
        "# Resolve paths from repo root\n"
        "cwd = Path(os.getcwd())\n"
        "def _find_base(path):\n"
        "    for p in [path] + list(path.parents):\n"
        "        if (p / 'resources' / 'market').exists(): return p\n"
        "    return path\n"
        "base = _find_base(cwd)\n"
        "sys.path.insert(0, str(base))\n"
        "for k, v in list(PATHS.items()):\n"
        "    p = Path(v)\n"
        "    if not p.is_absolute():\n"
        "        c = base / v\n"
        "        PATHS[k] = str(c.resolve() if c.exists() else (cwd / v).resolve())\n"
        "\n"
        "COLORS = {\n"
        "    'wind_offshore': '#1f77b4', 'wind_onshore': '#aec7e8',\n"
        "    'solar_pv': '#ffdd57', 'nuclear': '#9467bd',\n"
        "    'CCGT': '#e86414', 'OCGT': '#fd8d3c',\n"
        "    'coal': '#636363', 'oil': '#8c6d31',\n"
        "    'Battery': '#2ca02c', 'Pumped Storage Hydroelectricity': '#17becf',\n"
        "    'large_hydro': '#98df8a', 'small_hydro': '#c7e9c0',\n"
        "    'load_shedding': '#d62728', 'waste_to_energy': '#8c564b',\n"
        "    'landfill_gas': '#bcbd22', 'biogas': '#6baed6',\n"
        "    'biomass': '#74c476', 'marine': '#3182bd',\n"
        "    'advanced_biofuel': '#d6616b', 'sewage_gas': '#e7ba52',\n"
        "    'tidal_stream': '#3182bd', 'shoreline_wave': '#6baed6',\n"
        "    'EU_import': '#e377c2',\n"
        "    'interconnector': '#999999',\n"
        "}\n"
        "\n"
        "# Load data\n"
        "wp  = pd.read_csv(PATHS['wholesale_price'], index_col=0, parse_dates=True)\n"
        "gen = pd.read_csv(PATHS['wholesale_dispatch'], index_col=0, parse_dates=True)\n"
        "su  = pd.read_csv(PATHS['wholesale_storage'], index_col=0, parse_dates=True)\n"
        "mc  = pd.read_csv(PATHS['marginal_costs'])\n"
        "n   = pypsa.Network(PATHS['wholesale_network'])\n"
        "gen_carriers = n.generators['carrier']\n"
        "su_carriers  = n.storage_units['carrier'] if len(n.storage_units) else pd.Series(dtype=str)\n"
        "smp = wp['wholesale_price']\n"
        "\n"
        "print(f'Scenario : {SCENARIO}')\n"
        "print(f'Period   : {wp.index[0]}  to  {wp.index[-1]}  ({len(wp)} timesteps)')\n"
        "print(f'Generators: {len(gen.columns)}  |  Storage: {len(su.columns)}')\n"
        "print(f'Installed capacity: {n.generators.p_nom.sum()/1000:.1f} GW')"
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Merit Order
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md(
        "---\n"
        "## Merit Order\n"
        "\n"
        "Supply curve: each bar is one generator — width = capacity (GW), "
        "height = marginal cost (\u00a3/MWh), sorted cheapest-first. "
        "Interconnectors shown using their link capacity and mean ENTSO-E "
        "day-ahead price. "
        "Dashed black line = mean demand, dotted red = peak demand. "
        "Load shedding (\u00a36,000/MWh) excluded from y-axis."
    ))

    cells.append(_code(
        "MC_COL = 'marginal_cost'\n"
        "\n"
        "# ── Add interconnectors (EU_supply generators) to the merit order ──\n"
        "# They are added after apply_marginal_costs, so not in the CSV.\n"
        "# Capacity = sum of IC link p_nom feeding each external bus (actual import limit).\n"
        "# MC = mean of time-varying ENTSO-E price, or static marginal_cost.\n"
        "eu_gens = n.generators[n.generators.carrier == 'EU_import']\n"
        "ic_rows = []\n"
        "for gen_name, row in eu_gens.iterrows():\n"
        "    ext_bus = row['bus']\n"
        "    # Sum link capacities that connect TO this external bus\n"
        "    links_to = n.links[n.links.bus1 == ext_bus]\n"
        "    links_from = n.links[n.links.bus0 == ext_bus]\n"
        "    cap_mw = links_to.p_nom.sum() + links_from.p_nom.sum()\n"
        "    if cap_mw == 0:\n"
        "        cap_mw = row['p_nom']  # fallback to generator p_nom\n"
        "    # Mean MC: time-varying if available, else static\n"
        "    if gen_name in n.generators_t.marginal_cost.columns:\n"
        "        avg_mc = n.generators_t.marginal_cost[gen_name].mean()\n"
        "    else:\n"
        "        avg_mc = row['marginal_cost']\n"
        "    ic_rows.append({\n"
        "        'generator': gen_name, 'carrier': 'EU_import',\n"
        "        'p_nom_MW': cap_mw, 'marginal_cost': avg_mc,\n"
        "    })\n"
        "ic_df = pd.DataFrame(ic_rows) if ic_rows else pd.DataFrame(\n"
        "    columns=['generator', 'carrier', 'p_nom_MW', 'marginal_cost'])\n"
        "\n"
        "mo_df = pd.concat([mc, ic_df], ignore_index=True)\n"
        "mo_df = mo_df[mo_df['carrier'] != 'load_shedding'].copy()\n"
        "mo_df['p_nom_gw'] = mo_df['p_nom_MW'] / 1000\n"
        "mo_df = mo_df.sort_values(MC_COL).reset_index(drop=True)\n"
        "mo_df['cum_end'] = mo_df['p_nom_gw'].cumsum()\n"
        "mo_df['cum_start'] = mo_df['cum_end'] - mo_df['p_nom_gw']\n"
        "\n"
        "demand_ts = n.loads_t.p_set.sum(axis=1)\n"
        "mean_demand_gw = demand_ts.mean() / 1000\n"
        "peak_demand_gw = demand_ts.max() / 1000\n"
        "mean_price = smp.mean()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(16, 8))\n"
        f"fig.suptitle(f'Merit Order (Supply Curve) \\u2014 {scenario}', fontsize=13)\n"
        "\n"
        "x_centers = (mo_df['cum_start'] + mo_df['cum_end']) / 2\n"
        "bar_colors = [COLORS.get(c, '#aaa') for c in mo_df['carrier']]\n"
        "ax.bar(x_centers.values, mo_df[MC_COL].values,\n"
        "       width=mo_df['p_nom_gw'].values,\n"
        "       color=bar_colors, edgecolor='none', align='center')\n"
        "ax.axhline(0, color='black', linewidth=0.8, alpha=0.6)\n"
        "ax.axvline(mean_demand_gw, color='black', linestyle='--', linewidth=2.0)\n"
        "ax.axvline(peak_demand_gw, color='#d62728', linestyle=':', linewidth=1.8)\n"
        "\n"
        "ymin_val = mo_df[MC_COL].min()\n"
        "ymin = ymin_val * 1.15 if ymin_val < 0 else -5\n"
        "ymax = max(mo_df[MC_COL].quantile(0.99) * 1.15, 250)\n"
        "ax.set_ylim(ymin, ymax)\n"
        "ax.set_xlim(0, mo_df['cum_end'].max() * 1.01)\n"
        "ax.set_xlabel('Cumulative Installed Capacity (GW)', fontsize=11)\n"
        "ax.set_ylabel('Marginal Cost (\\u00a3/MWh)', fontsize=11)\n"
        "ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'\\u00a3{x:.0f}'))\n"
        "\n"
        "seen = mo_df.groupby('carrier')['p_nom_gw'].sum().sort_values(ascending=False)\n"
        "legend_handles = [\n"
        "    Patch(facecolor=COLORS.get(c, '#aaa'), label=f'{c}  ({v:.1f} GW)')\n"
        "    for c, v in seen.items()\n"
        "]\n"
        "legend_handles += [\n"
        "    plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2,\n"
        "               label=f'Mean demand: {mean_demand_gw:.1f} GW'),\n"
        "    plt.Line2D([0], [0], color='#d62728', linestyle=':', linewidth=1.8,\n"
        "               label=f'Peak demand: {peak_demand_gw:.1f} GW'),\n"
        "]\n"
        "ax.legend(handles=legend_handles, loc='upper left', fontsize=8,\n"
        "          ncol=2, framealpha=0.9)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "neg_mc = mo_df[mo_df[MC_COL] < 0]\n"
        "zero_mc = mo_df[mo_df[MC_COL] == 0]\n"
        "pos_mc  = mo_df[mo_df[MC_COL] > 0]\n"
        "print(f'Total capacity (excl. VOLL): {mo_df[\"p_nom_gw\"].sum():.1f} GW ({len(mo_df)} units)')\n"
        "print(f'  Negative MC : {neg_mc[\"p_nom_gw\"].sum():.1f} GW ({len(neg_mc)} units)')\n"
        "print(f'  Zero MC     : {zero_mc[\"p_nom_gw\"].sum():.1f} GW ({len(zero_mc)} units)')\n"
        "print(f'  Positive MC : {pos_mc[\"p_nom_gw\"].sum():.1f} GW ({len(pos_mc)} units)')\n"
        "print(f'Mean demand   : {mean_demand_gw:.1f} GW  |  Peak: {peak_demand_gw:.1f} GW')\n"
        "print(f'Mean SMP      : \\u00a3{mean_price:.0f}/MWh')\n"
        "if len(ic_df):\n"
        "    print(f'\\nInterconnectors ({len(ic_df)} links):')\n"
        "    for _, r in ic_df.iterrows():\n"
        "        print(f'  {r[\"generator\"]:40s}  {r[\"p_nom_MW\"]/1000:.1f} GW  \\u00a3{r[\"marginal_cost\"]:.0f}/MWh')"
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Dispatch Stack
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md(
        "---\n"
        "## Dispatch Stack\n"
        "\n"
        "Generation by carrier at each timestep. Carriers stacked cheapest (bottom) "
        "to most expensive (top). Black line = demand."
    ))

    cells.append(_code(
        "def _by_carrier(dispatch_df, carrier_series):\n"
        "    t = dispatch_df.T.copy()\n"
        "    t['carrier'] = carrier_series.reindex(t.index)\n"
        "    return t.groupby('carrier').sum().T\n"
        "\n"
        "gen_stack = _by_carrier(gen, gen_carriers)\n"
        "su_stack  = _by_carrier(su, su_carriers) if len(su_carriers) else pd.DataFrame(index=gen.index)\n"
        "\n"
        "full_stack = gen_stack.copy()\n"
        "for col in su_stack.columns:\n"
        "    if col in full_stack.columns:\n"
        "        full_stack[col] = full_stack[col].add(su_stack[col], fill_value=0)\n"
        "    else:\n"
        "        full_stack[col] = su_stack[col]\n"
        "\n"
        "# Sort by mean MC\n"
        "mc_order = mc.groupby('carrier')['marginal_cost'].mean().sort_values()\n"
        "ordered_cols = [c for c in mc_order.index if c in full_stack.columns]\n"
        "ordered_cols += [c for c in full_stack.columns if c not in ordered_cols]\n"
        "full_stack = full_stack[ordered_cols]\n"
        "\n"
        "pos_stack = full_stack.clip(lower=0)\n"
        "neg_stack = full_stack.clip(upper=0)\n"
        "demand_ts_plot = n.loads_t.p_set.sum(axis=1).reindex(gen.index)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(14, 6))\n"
        f"fig.suptitle(f'Dispatch Stack \\u2014 {scenario}', fontsize=13)\n"
        "x = range(len(pos_stack))\n"
        "bottom = np.zeros(len(pos_stack))\n"
        "for col in pos_stack.columns:\n"
        "    if pos_stack[col].sum() < 1: continue\n"
        "    ax.bar(x, pos_stack[col].values / 1000, bottom=bottom / 1000,\n"
        "           color=COLORS.get(col, '#999'), label=col, width=1.0, linewidth=0)\n"
        "    bottom += pos_stack[col].values\n"
        "\n"
        "neg_bottom = np.zeros(len(neg_stack))\n"
        "for col in neg_stack.columns:\n"
        "    if neg_stack[col].min() > -1: continue\n"
        "    ax.bar(x, neg_stack[col].values / 1000, bottom=neg_bottom / 1000,\n"
        "           color=COLORS.get(col, '#ccc'), width=1.0, linewidth=0, alpha=0.6)\n"
        "    neg_bottom += neg_stack[col].values\n"
        "\n"
        "ax.plot(x, demand_ts_plot.values / 1000, 'k-', linewidth=1.5,\n"
        "        label='Demand', zorder=5)\n"
        "n_ticks = min(12, len(pos_stack))\n"
        "step = max(1, len(pos_stack) // n_ticks)\n"
        "tick_pos = list(range(0, len(pos_stack), step))\n"
        "ax.set_xticks(tick_pos)\n"
        "ax.set_xticklabels([str(pos_stack.index[i])[:16] for i in tick_pos],\n"
        "                    rotation=30, ha='right', fontsize=8)\n"
        "ax.set_ylabel('Power (GW)')\n"
        "ax.axhline(0, color='black', linewidth=0.8)\n"
        "handles, labels = ax.get_legend_handles_labels()\n"
        "by_label = dict(zip(labels, handles))\n"
        "ax.legend(by_label.values(), by_label.keys(), loc='upper right',\n"
        "          fontsize=7, ncol=2)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Wholesale Price vs Market
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(_md(
        "---\n"
        "## Wholesale Price vs Market\n"
        "\n"
        "The model SMP (blue) compared against **ELEXON Market Index Data (MID)** "
        "real wholesale price (orange). For historical scenarios (\u22642024), MID data "
        "is fetched from the ELEXON API.\n"
        "\n"
        "| Metric | Meaning |\n"
        "|--------|---------|\n"
        "| **Bias** | mean(SMP \u2212 MID): positive = model over-predicts |\n"
        "| **r** | correlation: how well model tracks price shape |\n"
        "| **RMSE** | root mean square error |"
    ))

    cells.append(_code(
        "# Fetch MID prices for comparison\n"
        "mid = None\n"
        "if MODELLED_YEAR <= 2024 and SOLVE_START:\n"
        "    try:\n"
        "        from scripts.market.elexon_data import fetch_mid_prices_bulk\n"
        "        start_str = SOLVE_START[:10]\n"
        "        end_str   = SOLVE_END[:10]\n"
        "        _mid_raw = fetch_mid_prices_bulk(start_str, end_str)\n"
        "        # MID is half-hourly; resample to hourly to match model timestep.\n"
        "        # Averaging the two 30-min SPs per hour is more accurate than reindex,\n"
        "        # and avoids duplicate-axis errors from BST\\u2192GMT clock-change days.\n"
        "        mid = _mid_raw.resample('h').mean().reindex(smp.index)\n"
        "        print(f'Fetched {mid.notna().sum()} MID price points')\n"
        "    except Exception as e:\n"
        "        print(f'Could not fetch MID prices: {e}')\n"
        "\n"
        "if mid is not None:\n"
        "    # Guard: ensure index is aligned with smp before boolean indexing\n"
        "    if not mid.index.equals(smp.index):\n"
        "        mid = mid.reindex(smp.index)\n"
        "    mask = mid.notna()\n"
        "    n_pts = mask.sum()\n"
        "    if n_pts >= 10:\n"
        "        diff = smp[mask] - mid[mask]\n"
        "        bias = diff.mean()\n"
        "        r = np.corrcoef(smp[mask], mid[mask])[0, 1]\n"
        "        rmse = np.sqrt((diff ** 2).mean())\n"
        "        mae  = diff.abs().mean()\n"
        "        print(f'Bias : {bias:+.1f} \\u00a3/MWh')\n"
        "        print(f'r    : {r:.3f}')\n"
        "        print(f'RMSE : {rmse:.1f} \\u00a3/MWh')\n"
        "        print(f'MAE  : {mae:.1f} \\u00a3/MWh')\n"
        "        print(f'n    : {n_pts}')\n"
        "    else:\n"
        "        print(f'Only {n_pts} matched points')\n"
        "        bias = r = rmse = mae = None\n"
        "else:\n"
        "    bias = r = rmse = mae = mask = None\n"
        "    print('No MID data (future scenario or fetch failed)')"
    ))

    cells.append(_code(
        "fig, axes = plt.subplots(2, 2, figsize=(14, 9),\n"
        "                         gridspec_kw={'height_ratios': [2, 1]})\n"
        f"fig.suptitle(f'Wholesale Price vs Market \\u2014 {scenario}', fontsize=13)\n"
        "\n"
        "# ── Top-left: time series ──────────────────────────────────────────\n"
        "ax = axes[0, 0]\n"
        "ax.plot(smp.index, smp.values, 'o-', ms=3, color='#1f77b4',\n"
        "        label=f'Model SMP (mean \\u00a3{smp.mean():.0f}/MWh)', zorder=3)\n"
        "if mid is not None and mask is not None and mask.sum() > 0:\n"
        "    ax.plot(mid.index, mid.values, 's-', ms=3, color='#ff7f0e',\n"
        "            label=f'ELEXON MID (mean \\u00a3{mid[mask].mean():.0f}/MWh)', zorder=2)\n"
        "    if bias is not None:\n"
        "        txt = f'bias = {bias:+.1f} \\u00a3/MWh\\nr = {r:.3f}\\nRMSE = {rmse:.1f}'\n"
        "        ax.text(0.98, 0.97, txt, transform=ax.transAxes, va='top', ha='right',\n"
        "                fontsize=9, family='monospace',\n"
        "                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))\n"
        "ax.set_ylabel('\\u00a3/MWh')\n"
        "ax.legend(fontsize=9)\n"
        "ax.tick_params(axis='x', labelrotation=30)\n"
        "\n"
        "# ── Top-right: scatter plot ────────────────────────────────────────\n"
        "ax2 = axes[0, 1]\n"
        "if mid is not None and mask is not None and mask.sum() >= 10:\n"
        "    ax2.scatter(mid[mask], smp[mask], s=20, alpha=0.5, color='#2ca02c')\n"
        "    lims = [min(mid[mask].min(), smp[mask].min()) - 10,\n"
        "            max(mid[mask].max(), smp[mask].max()) + 10]\n"
        "    ax2.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='1:1')\n"
        "    ax2.set_xlim(lims); ax2.set_ylim(lims)\n"
        "    ax2.set_xlabel('MID (\\u00a3/MWh)')\n"
        "    ax2.set_ylabel('Model SMP (\\u00a3/MWh)')\n"
        "    ax2.legend(fontsize=9)\n"
        "    ax2.set_title('SMP vs MID scatter')\n"
        "else:\n"
        "    ax2.text(0.5, 0.5, 'No MID data', transform=ax2.transAxes,\n"
        "             ha='center', va='center', fontsize=12, color='grey')\n"
        "    ax2.set_title('SMP vs MID scatter')\n"
        "\n"
        "# ── Bottom-left: price duration curve ──────────────────────────────\n"
        "ax3 = axes[1, 0]\n"
        "sorted_smp = smp.sort_values(ascending=False).reset_index(drop=True)\n"
        "ax3.fill_between(range(len(sorted_smp)), sorted_smp.values,\n"
        "                 alpha=0.3, color='#1f77b4')\n"
        "ax3.plot(range(len(sorted_smp)), sorted_smp.values, color='#1f77b4',\n"
        "         linewidth=1.5, label='Model SMP')\n"
        "if mid is not None and mask is not None and mask.sum() > 0:\n"
        "    sorted_mid = mid.dropna().sort_values(ascending=False).reset_index(drop=True)\n"
        "    ax3.fill_between(range(len(sorted_mid)), sorted_mid.values,\n"
        "                     alpha=0.3, color='#ff7f0e')\n"
        "    ax3.plot(range(len(sorted_mid)), sorted_mid.values, color='#ff7f0e',\n"
        "             linewidth=1.5, label='ELEXON MID')\n"
        "ax3.set_xlabel('Hours (ranked by descending price)')\n"
        "ax3.set_ylabel('\\u00a3/MWh')\n"
        "ax3.set_title('Price duration curve')\n"
        "ax3.legend(fontsize=9)\n"
        "\n"
        "# ── Bottom-right: error histogram ──────────────────────────────────\n"
        "ax4 = axes[1, 1]\n"
        "if mid is not None and mask is not None and mask.sum() >= 10:\n"
        "    err = smp[mask] - mid[mask]\n"
        "    ax4.hist(err, bins=25, color='#9467bd', edgecolor='white', alpha=0.8)\n"
        "    ax4.axvline(0, color='black', linewidth=1)\n"
        "    ax4.axvline(err.mean(), color='#d62728', linestyle='--',\n"
        "               label=f'Mean bias: {err.mean():+.1f}')\n"
        "    ax4.set_xlabel('SMP \\u2212 MID (\\u00a3/MWh)')\n"
        "    ax4.set_ylabel('Count')\n"
        "    ax4.set_title('Error distribution')\n"
        "    ax4.legend(fontsize=9)\n"
        "else:\n"
        "    ax4.text(0.5, 0.5, 'No MID data', transform=ax4.transAxes,\n"
        "             ha='center', va='center', fontsize=12, color='grey')\n"
        "    ax4.set_title('Error distribution')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── Assemble ──────────────────────────────────────────────────────────────
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "pypsa-gb",
                "language": "python",
                "name": "pypsa-gb",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
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
            "wholesale_network":      snakemake.input.wholesale_network,
            "wholesale_dispatch_csv": snakemake.input.wholesale_dispatch_csv,
            "wholesale_storage_csv":  snakemake.input.wholesale_storage_csv,
            "wholesale_links_csv":    snakemake.input.wholesale_links_csv,
            "wholesale_price_csv":    snakemake.input.wholesale_price_csv,
            "marginal_costs_csv":     snakemake.input.marginal_costs_csv,
        }
        output_path = snakemake.output.notebook
        scenario_config = getattr(snakemake.params, "scenario_config", {})
    except NameError:
        import sys
        if len(sys.argv) < 3:
            print("Usage: python generate_wholesale_notebook.py <scenario> <output.ipynb>")
            sys.exit(1)
        scenario = sys.argv[1]
        base_market = f"resources/market/{scenario}"
        inputs = {
            "wholesale_network":      f"{base_market}_wholesale.nc",
            "wholesale_dispatch_csv": f"{base_market}_wholesale_dispatch.csv",
            "wholesale_storage_csv":  f"{base_market}_wholesale_storage.csv",
            "wholesale_links_csv":    f"{base_market}_wholesale_links.csv",
            "wholesale_price_csv":    f"{base_market}_wholesale_price.csv",
            "marginal_costs_csv":     f"resources/generators/{scenario}_marginal_costs_breakdown.csv",
        }
        output_path = sys.argv[2]
        scenario_config = {}

    logger.info(f"Building wholesale notebook for scenario: {scenario}")

    nb = build_notebook(scenario, inputs, scenario_config)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)

    logger.info(f"Notebook written to {output_path}")


main()
