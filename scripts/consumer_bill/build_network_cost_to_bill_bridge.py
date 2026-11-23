"""Build a 2020 network-cost bridge without treating PyPSA flows as tariffs.

The bridge keeps the published Ofgem allowance as the regulatory source. It
uses PyPSA only where it has a valid role: allocating volumetric TNUoS/DUoS
allowances across delivery periods and indexing balancing activity. Fixed DUoS
charges remain account/year based, and losses require a physical-network loss
signal before they can leave their published baseline.
"""


import argparse
import html
import json
from pathlib import Path

import pandas as pd


COMPONENTS = {
    "TNUoS": "Ofgem Annex 3 — TNUoS",
    "DUoS unit charges": "Ofgem Annex 3 — DUoS unit charges",
    "DUoS fixed charges": "Ofgem Annex 3 — DUoS fixed charges",
}
RECOVERY_COMPONENTS = {
    "Ofgem Annex 3 — BSUoS",
    "Imbalance uplift",
    "GB-average regional electricity losses",
}


def _published(register: pd.DataFrame, component: str) -> float:
    values = register.loc[
        register["component"].eq(component), "annual_gbp_per_3100kwh_ex_vat"
    ]
    if len(values) != 1:
        raise ValueError(f"Expected one benchmark component: {component}")
    return float(pd.to_numeric(values.iloc[0], errors="raise"))


def _recovery_row(recovery: pd.DataFrame, component: str) -> pd.Series | None:
    matched = recovery.loc[recovery["component"].eq(component)]
    if len(matched) > 1:
        raise ValueError(f"Recovery bridge has duplicate component: {component}")
    return matched.iloc[0] if len(matched) else None


def build_network_cost_bridge(
    register: pd.DataFrame,
    network_components: pd.DataFrame,
    wholesale: pd.DataFrame,
    recovery: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Return a component-level same-year network-cost allocation bridge."""
    for frame, required, name in [
        (register, {"component", "annual_gbp_per_3100kwh_ex_vat"}, "benchmark register"),
        (network_components, {"cap_period", "component", "annual_gbp_per_3100kwh_ex_vat", "calendar_weight_2020"}, "network component input"),
        (wholesale, {"cap_period", "modelled_delivery_demand_share"}, "wholesale impact"),
        (recovery, {"component", "candidate_gbp_per_3100kwh_ex_vat", "evidence_status"}, "recovery bridge"),
    ]:
        if missing := required.difference(frame.columns):
            raise ValueError(f"{name} is missing: {sorted(missing)}")
    demand_shares = wholesale[["cap_period", "modelled_delivery_demand_share"]].copy()
    demand_shares["modelled_delivery_demand_share"] = pd.to_numeric(
        demand_shares["modelled_delivery_demand_share"], errors="raise"
    )
    if demand_shares["cap_period"].duplicated().any() or abs(demand_shares["modelled_delivery_demand_share"].sum() - 1.0) > 1e-8:
        raise ValueError("Wholesale impact requires unique cap periods whose demand shares sum to one")

    rows: list[dict] = []
    for source, bill_component in COMPONENTS.items():
        period = network_components.loc[network_components["component"].eq(source)].copy()
        period = period.merge(demand_shares, on="cap_period", how="inner", validate="one_to_one")
        if len(period) != len(demand_shares):
            raise ValueError(f"Network input does not cover all cap periods for {source}")
        period["annual_gbp_per_3100kwh_ex_vat"] = pd.to_numeric(
            period["annual_gbp_per_3100kwh_ex_vat"], errors="raise"
        )
        published = _published(register, bill_component)
        if source == "DUoS fixed charges":
            candidate = float((period["annual_gbp_per_3100kwh_ex_vat"] * period["calendar_weight_2020"]).sum())
            method = "Published period fixed charge annualised by calendar days; this account/meter charge is not allocated using PyPSA system demand"
            signal = "calendar-day customer-year allocation"
        else:
            candidate = float((period["annual_gbp_per_3100kwh_ex_vat"] * period["modelled_delivery_demand_share"]).sum())
            method = "Published period allowance allocated using PyPSA modelled delivery-period demand share; tariff/allowed-revenue setting remains external"
            signal = "PyPSA delivery-period demand share"
        rows.append({
            "component": bill_component,
            "published_allowance_gbp_per_3100kwh_ex_vat": published,
            "candidate_allowance_gbp_per_3100kwh_ex_vat": candidate,
            "difference_gbp_per_3100kwh_ex_vat": candidate - published,
            "bridge": "Transmission charging allocation" if source == "TNUoS" else "Distribution charging allocation",
            "modelled_signal": signal,
            "recovery_status": "candidate_ready_for_same_year_delivery_allocation",
            "same_year_backtest_allowed": True,
            "model_basis": method,
        })

    for component in sorted(RECOVERY_COMPONENTS):
        if component not in set(register["component"]):
            continue
        source = _recovery_row(recovery, component)
        published = _published(register, component)
        candidate = None if source is None else pd.to_numeric(
            pd.Series([source["candidate_gbp_per_3100kwh_ex_vat"]]), errors="coerce"
        ).iloc[0]
        status = "input_required" if source is None else str(source["evidence_status"])
        is_balancing = component in {"Ofgem Annex 3 — BSUoS", "Imbalance uplift"}
        allowed = bool(is_balancing and candidate is not None and status.startswith("calibrated_same_year_volume_index"))
        if component == "GB-average regional electricity losses":
            allowed = bool(candidate is not None and status.startswith("candidate_ready"))
        rows.append({
            "component": component,
            "published_allowance_gbp_per_3100kwh_ex_vat": published,
            "candidate_allowance_gbp_per_3100kwh_ex_vat": candidate,
            "difference_gbp_per_3100kwh_ex_vat": None if pd.isna(candidate) else float(candidate - published),
            "bridge": "Balancing / BSUoS activity recovery" if is_balancing else "Loss-factor recovery",
            "modelled_signal": None if source is None else source.get("modelled_signal_unit"),
            "recovery_status": status,
            "same_year_backtest_allowed": allowed,
            "model_basis": None if source is None else source.get("method_note"),
        })

    result = pd.DataFrame(rows).sort_values("component").reset_index(drop=True)
    summary = {
        "purpose": "2020 consumer-bill network-cost bridge",
        "same_year_delivery_allocation_components": list(COMPONENTS.values()),
        "balancing_components": ["Ofgem Annex 3 — BSUoS", "Imbalance uplift"],
        "losses_treatment": "Published baseline retained unless a separately validated physical-network loss bridge is available",
        "same_year_backtest_candidate_total_gbp_per_3100kwh_ex_vat": float(
            result.loc[result["same_year_backtest_allowed"], "candidate_allowance_gbp_per_3100kwh_ex_vat"].sum()
        ),
        "interpretation": "TNUoS and DUoS are regulatory charging allowances, not outputs of PyPSA flows. This bridge makes only a same-year delivery/allocation calculation; allowed revenue and tariff setting remain external inputs.",
    }
    return result, summary


def _html(rows: pd.DataFrame, summary: dict) -> str:
    body = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.component))}</td><td>£{row.published_allowance_gbp_per_3100kwh_ex_vat:,.2f}</td>"
        f"<td>{'' if pd.isna(row.candidate_allowance_gbp_per_3100kwh_ex_vat) else f'£{row.candidate_allowance_gbp_per_3100kwh_ex_vat:,.2f}'}</td>"
        f"<td>{html.escape(str(row.recovery_status))}</td></tr>"
        for row in rows.itertuples(index=False)
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>2020 network-cost bridge</title><style>body{{font-family:Arial;max-width:1000px;margin:32px auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#f3f6f9}}</style></head><body><h1>2020 network-cost bridge</h1><p>{html.escape(summary['interpretation'])}</p><table><thead><tr><th>Component</th><th>Published</th><th>Candidate</th><th>Status</th></tr></thead><tbody>{body}</tbody></table></body></html>"""


def write_bridge(register_path, network_components_path, wholesale_path, recovery_path, output_csv, output_summary, output_html):
    rows, summary = build_network_cost_bridge(
        pd.read_csv(register_path), pd.read_csv(network_components_path),
        pd.read_csv(wholesale_path), pd.read_csv(recovery_path),
    )
    for output in (output_csv, output_summary, output_html):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(output_html).write_text(_html(rows, summary), encoding="utf-8")


if "snakemake" in globals():
    write_bridge(
        snakemake.input.register_csv, snakemake.input.network_components_csv,
        snakemake.input.wholesale_impact_csv, snakemake.input.recovery_bridges_csv,
        snakemake.output.bridge_csv, snakemake.output.summary_json, snakemake.output.report_html,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", required=True); parser.add_argument("--network-components", required=True)
    parser.add_argument("--wholesale-impact", required=True); parser.add_argument("--recovery-bridges", required=True)
    parser.add_argument("--output-csv", required=True); parser.add_argument("--output-summary", required=True); parser.add_argument("--output-html", required=True)
    args = parser.parse_args()
    write_bridge(args.register, args.network_components, args.wholesale_impact, args.recovery_bridges,
                 args.output_csv, args.output_summary, args.output_html)
