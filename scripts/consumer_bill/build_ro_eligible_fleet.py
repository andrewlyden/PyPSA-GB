"""Estimate the 2030 Renewables Obligation scheme cost from the Ofgem register.

The RO line in the CP30_2030_cp30_network bill is an external GBP17, which implicitly assumes
the 75% Exchequer relief on domestic supply continues. Autumn Budget 2025 funds
75% of the domestic share of the legacy RO from taxation for three years from
April 2026, returning it to bills in 2029-30, so calendar 2030 would be the
first full year of gross recovery: the line is a 55-75 GBP swing that is
otherwise invisible. The two relief cases in the scheme parameters bracket it.

The fleet (which stations still hold accreditation) comes from the Ofgem
register; the energy comes from the model. Since the CP30 brownfield rebuild
the 2030 networks carry the named REPD operating fleet, with ROC-labelled units
(``support_type == "ROC"``) for offshore and onshore wind and solar, so the load
factor is taken from those units' own dispatched output wherever they exist
(26 Sep 2026). A carrier average undercounted ROCs by about a fifth: ROC units
bid minus their certificate value, so the wholesale and balancing stages rarely
curtail them, while the carrier average is dragged down by CfD and merchant
units curtailed at zero (2030 as-built run: offshore 0.454 ROC units against a
0.376 carrier average, onshore 0.289 against 0.175; 62.5m ROCs rose to about
78m). Route, labelled `pypsa_informed_fleet_register`:

1. take the Ofgem RO accredited-stations register and weight each station by
   the share of calendar 2030 its 20-year accreditation still runs (the bill is
   calendar 2030, which straddles obligation years 2029/30 and 2030/31);
2. multiply that capacity by the load factor of the carrier's ROC-labelled
   units in the 2030 balancing (physical) dispatch, falling back to the
   carrier average where the model holds too little ROC-labelled capacity
   (``load_factor_basis``), so the energy is model-informed;
3. multiply by ROC banding to get certificate volume, and by the notional ROC
   value (buy-out plus recycle) to get gross scheme cost;
4. divide by obligation-eligible supply and express per benchmark customer,
   under two named Exchequer-relief cases.

The buy-out price needs no inflation forecast. It is RPI-indexed every year, so
holding it flat in real RPI terms and restating the last sourced value in report
prices gives the 2030/31 level directly.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.public_ownership.model_year import (  # noqa: E402
    DEFAULT_MODEL_YEAR, calendar_period, fiscal_year_label, recovery_model_year,
)


#: The bill is calendar 2030 (28 Sep 2026): the model dispatch is January to
#: December 2030, and calendar 2030 straddles obligation years 2029/30 (January
#: to March) and 2030/31. The buy-out is flat in real RPI terms, so the period
#: only decides which accreditations count, and for how much of the year.
def bill_period(model_year: int) -> tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """The bill's obligation period for a model year (item Y4).

    Args:
        model_year: the calendar year the bill is for (the recovery file's ``forecast_year``).

    Returns:
        ``(label, period start, period end, 1 April of the model year)``.
    """
    start, end = calendar_period(model_year)
    return f"calendar {int(model_year)}", start, end, pd.Timestamp(f"{int(model_year)}-04-01")


# The default year's period, kept for callers that import these names; the bill
# takes its own year from the recovery-demand file (item Y4).
OBLIGATION_YEAR, PERIOD_START, PERIOD_END, _DEFAULT_OBLIGATION_YEAR_START = bill_period(DEFAULT_MODEL_YEAR)
# Load factors come from the ROC-labelled units of each carrier when the model
# holds at least this share of the carrier's register-eligible capacity as
# ROC-labelled units; below it the labelled units are too few to stand for the
# fleet and the carrier average is used. The 2030 networks hold 74% (offshore),
# 80% (onshore) and 136% (solar) of the register capacity as ROC units, and none
# for biomass, hydro or marine. Overridable in the scheme parameters under
# ``load_factor_basis``; the basis used is recorded per carrier (load_factor_source).
DEFAULT_LOAD_FACTOR_BASIS = {"mode": "roc_labelled_units", "min_labelled_share_of_register": 0.25}
LOAD_FACTOR_BASIS_MODES = ("roc_labelled_units", "carrier_average")
# Kept for callers of read_eligible_fleet that still pass a single start date;
# the bill's own route weights each station over PERIOD_START..PERIOD_END.
OBLIGATION_YEAR_START = _DEFAULT_OBLIGATION_YEAR_START
# The RO closes to all generation support on 31 March 2037.
RO_SCHEME_END = pd.Timestamp("2037-03-31")
ROC_ACCREDITATION_YEARS = 20

# Ofgem register technology -> PyPSA carrier, for load factors and banding.
# Register technologies with no CP30 carrier equivalent are mapped to the
# closest represented carrier and reported separately.
REGISTER_TECHNOLOGY_TO_CARRIER = {
    "Onshore wind": "wind_onshore",
    "Wind with a DNC of 50kW or less": "wind_onshore",
    "Offshore wind": "wind_offshore",
    "Offshore wind - demonstration turbines": "wind_offshore",
    "Offshore wind - floating turbines": "wind_offshore",
    "Photovoltaic": "solar_pv",
    "Photovoltaic with a DNC of 50kW or less": "solar_pv",
    "Fuelled": "biomass",
    "Fuelled with a DNC of 50kW or less": "biomass",
    "Landfill gas": "biomass",
    "Sewage gas": "biomass",
    "Hydro with a DNC of 20MW or less": "large_hydro",
    "Hydro with a DNC of 50kW or less": "large_hydro",
    "Hydro with a DNC greater than 20 MW": "large_hydro",
    "Micro hydro": "large_hydro",
    "Hydro": "large_hydro",
    "Tidal stream": "marine",
    "Wave power": "marine",
    "Tidal barrage and tidal lagoon": "marine",
    "Tidal flow": "marine",
    "Geopressure": "biomass",
    "Co-firing of biomass with fossil fuel": "biomass",
    "Co-firing of energy crops": "biomass",
    "Dedicated biomass with CHP": "biomass",
    "Dedicated energy crops": "biomass",
    "Microgeneration": "solar_pv",
}


def _dnc_mw(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce") / 1000.0


def accredited_share_of_period(effective, expiry, start=PERIOD_START, end=PERIOD_END) -> pd.Series:
    """The share of ``start``..``end`` each accreditation holds: 0 to 1, by day.

    A station whose accreditation ends in March 2030 still earns ROCs for a
    quarter of calendar 2030; one ending in October earns them for three
    quarters. Counting either wholly in or wholly out, by a single start date,
    mis-states the calendar-year certificate volume.
    """
    effective = pd.to_datetime(pd.Series(effective))
    expiry = pd.to_datetime(pd.Series(expiry))
    period = (end - start).days
    overlap = (expiry.clip(upper=end) - effective.clip(lower=start)).dt.days.clip(lower=0)
    return (overlap / period).clip(upper=1.0)


def read_eligible_fleet(register_path, obligation_year_start=OBLIGATION_YEAR_START,
                        accreditation_years=ROC_ACCREDITATION_YEARS,
                        period=(PERIOD_START, PERIOD_END)) -> pd.DataFrame:
    """Return the register with each station's share of the bill period it holds accreditation.

    ``eligible`` is a share above zero. ``accredited_share`` weights the
    station's capacity over ``period`` (calendar 2030 for the bill); with
    ``period=None`` the old single-date rule is kept (share 1 if the
    accreditation still runs on ``obligation_year_start``, else 0).
    """
    frame = pd.read_csv(register_path)
    required = {"Accreditation reference", "Station name", "Station DNC in kW",
                "Technology", "Effective date"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Ofgem RO register is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["capacity_mw"] = _dnc_mw(frame["Station DNC in kW"])
    frame["effective_date"] = pd.to_datetime(
        frame["Effective date"], format="%d/%m/%Y", errors="coerce"
    )
    if frame["effective_date"].isna().any():
        raise ValueError("Ofgem RO register has unparseable accreditation effective dates")
    if frame["capacity_mw"].isna().any():
        raise ValueError("Ofgem RO register has unparseable station DNC values")

    # 20 years from accreditation, and never past the scheme's own end date.
    frame["roc_expiry"] = frame["effective_date"] + pd.DateOffset(years=accreditation_years)
    frame["roc_expiry_capped"] = frame["roc_expiry"].clip(upper=RO_SCHEME_END)
    if period is None:
        frame["accredited_share"] = (frame["roc_expiry_capped"] > obligation_year_start).astype(float)
    else:
        frame["accredited_share"] = accredited_share_of_period(
            frame["effective_date"], frame["roc_expiry_capped"], *period).to_numpy()
    frame["eligible"] = frame["accredited_share"] > 0.0
    frame["carrier"] = frame["Technology"].map(REGISTER_TECHNOLOGY_TO_CARRIER)
    # Which RO jurisdiction the station is accredited in: Northern Ireland (NIRO)
    # stations cannot be on the GB network but their NIROCs are part of the UK
    # obligation GB suppliers pay for (see ``gb_share_of_uk_obligation``). A flag,
    # not an eligibility change, so the network's ROC labels are unaffected.
    country = (frame["Country"] if "Country" in frame.columns else pd.Series("", index=frame.index))
    frame["jurisdiction"] = np.where(country.fillna("").astype(str).str.strip().eq("Northern Ireland"), "NI", "GB")
    return frame


def gb_share_of_uk_obligation(parameters: dict) -> dict:
    """The share of the UK Renewables Obligation GB suppliers carry (NIRO question, 27 Sep 2026).

    ROCs trade in one UK-wide market: England and Wales ROCs, Scottish ROCs and
    Northern Ireland ROCs (NIROCs) are all presented against any supplier's
    obligation (Ofgem RO Guidance for Suppliers). DESNZ sets one UK total
    obligation from the ROCs expected to be issued to every UK station, NI
    included (Calculation B, plus 10% headroom), and divides it into a GB level
    and a lower NI level per MWh supplied. GB suppliers therefore pay for UK
    certificate issuance, NIROCs included, less the small part NI suppliers carry:
    their share is the GB obligation over the UK total.
    """
    block = parameters.get("uk_obligation_split")
    if not block:
        raise ValueError("ro scheme parameters need uk_obligation_split to apportion the UK obligation to GB")
    total = float(block["uk_total_obligation_rocs"])
    ni = float(block["ni_supply_twh"]) * 1e6 * float(block["ni_obligation_rocs_per_mwh"])
    if not 0 < ni < total:
        raise ValueError("uk_obligation_split: the NI obligation must be a positive part of the UK total")
    share = (total - ni) / total
    declared = block.get("gb_share_of_uk_obligation")
    if declared is not None and not np.isclose(float(declared), share, rtol=1e-6):
        raise ValueError("uk_obligation_split.gb_share_of_uk_obligation does not reconcile to its inputs")
    return {**block, "ni_obligation_rocs": ni, "gb_share_of_uk_obligation": share}


def read_certificate_evidence(certificates_path, schemes=("RO", "ROS")) -> pd.DataFrame:
    """Per accreditation: the last compliance year it was issued ROCs, and its banding.

    Ofgem's certificate register records the MWh behind each certificate for every
    issue, so its reciprocal is the station's actual banding (ROCs per MWh). The
    banding is the most common value in the station's latest issued compliance
    year. Great Britain issues under ``RO`` (England and Wales) and ``ROS``
    (Scotland); ``NIRO`` is Northern Ireland.
    """
    columns = ["Accreditation reference", "Scheme", "Compliance Period", "MWh per certificate"]
    certificates = pd.read_csv(certificates_path, usecols=columns, encoding="utf-8-sig")
    certificates = certificates[certificates["Scheme"].isin(schemes)].copy()
    certificates["compliance_year_start"] = pd.to_numeric(
        certificates["Compliance Period"].astype(str).str[:4], errors="coerce")
    mwh = pd.to_numeric(certificates["MWh per certificate"], errors="coerce")
    certificates["banding"] = (1.0 / mwh).where(mwh > 0).round(4)
    certificates = certificates.dropna(subset=["compliance_year_start"])
    last = certificates.groupby("Accreditation reference")["compliance_year_start"].max()
    latest = certificates.merge(last.rename("last_year"), left_on="Accreditation reference",
                                right_index=True)
    latest = latest[latest["compliance_year_start"].eq(latest["last_year"])]
    banding = latest.dropna(subset=["banding"]).groupby("Accreditation reference")["banding"].agg(
        lambda values: values.mode().iloc[0])
    return pd.DataFrame({"last_certificate_year": last, "observed_banding": banding})


def carrier_load_factors(networks, carriers, support_type=None) -> pd.DataFrame:
    """Compute annual load factors by carrier from the saved 2030 dispatch.

    ``support_type`` (e.g. ``"ROC"``) restricts each carrier to the generators
    carrying that support label; ``capacity_mw`` reports the largest monthly
    capacity so the caller can judge how much of the fleet the label covers.

    These must be the *balancing* (physical) networks, for two reasons. ROCs are
    issued on metered output, so wind that is constrained off earns none, and the
    CfD line and its curtailment compensation are both computed on balancing
    dispatch, so using it here keeps the support lines on one basis. The two
    stages can differ materially: constrained-off wind lowers the physical load
    factor below the copperplate one, while dispatchable carriers that the
    wholesale merit order barely calls can run harder once they are used to
    resolve constraints. The actual values are carried in the outputs
    (``pypsa_load_factor`` in the by-carrier CSV) rather than quoted here.
    """
    energy = {carrier: 0.0 for carrier in carriers}
    capability = {carrier: 0.0 for carrier in carriers}
    capacity = {carrier: 0.0 for carrier in carriers}
    for path in networks:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            snapshots = pd.DatetimeIndex(ds["snapshots_snapshot"].values)
            weights = pd.Series(ds["snapshots_generators"].values, index=snapshots, dtype=float)
            names = pd.Index(ds["generators_i"].values)
            carrier_of = pd.Series(ds["generators_carrier"].values, index=names)
            if support_type is not None:
                if "generators_support_type" not in ds:
                    raise ValueError(f"{path}: no generators_support_type to select {support_type!r} units by")
                support = pd.Series(ds["generators_support_type"].values, index=names).fillna("").astype(str)
                carrier_of = carrier_of.where(support.eq(support_type))
            p_nom = pd.Series(ds["generators_p_nom"].values, index=names, dtype=float)
            available = pd.Index(ds["generators_t_p_i"].values)
            hours = float(weights.sum())
            # Read the whole dispatch table once. A label ``.sel`` on the
            # NetCDF-backed array costs ~0.14 s per generator (hours for the
            # 11k-generator fleet); the full read takes well under a second.
            all_dispatch = ds["generators_t_p"].to_pandas()
            all_dispatch.index = snapshots
            # Pro rata zonal turn-down (20 Oct 2026): the month's basis.
            from scripts.market.turn_down_allocation import balancing_dispatch_in_force

            all_dispatch = balancing_dispatch_in_force(all_dispatch, path)
            for carrier in carriers:
                members = carrier_of.index[carrier_of.eq(carrier)].intersection(available)
                if not len(members):
                    continue
                capacity[carrier] = max(capacity[carrier], float(p_nom.reindex(members).sum()))
                dispatch = all_dispatch[members]
                energy[carrier] += float(dispatch.mul(weights, axis=0).to_numpy().sum())
                capability[carrier] += float(p_nom.reindex(members).sum()) * hours
    rows = []
    for carrier in carriers:
        if capability[carrier] <= 0:
            rows.append({"carrier": carrier, "load_factor": None,
                         "modelled_energy_mwh": energy[carrier], "capacity_mw": 0.0})
            continue
        rows.append({"carrier": carrier,
                     "load_factor": energy[carrier] / capability[carrier],
                     "modelled_energy_mwh": energy[carrier], "capacity_mw": capacity[carrier]})
    return pd.DataFrame(rows)


def notional_roc_value(parameters: dict, rpi: dict, model_year: int = DEFAULT_MODEL_YEAR) -> dict:
    """Restate the last sourced notional ROC value in report prices.

    The buy-out price is RPI-indexed annually, so in real RPI terms it is flat.
    Restating the sourced value at the report price year therefore gives the
    2030/31 level without any inflation forecast.
    """
    base = parameters["base_obligation_year"]
    factor = float(rpi["to_index"]) / float(rpi["from_index"])
    if int(rpi["from_price_year"]) != int(base["rpi_anchor_year"]):
        raise ValueError("RPI restatement does not start from the sourced buy-out anchor year")
    implied = float(rpi.get("factor", factor))
    if not np.isclose(implied, factor, rtol=1e-9):
        raise ValueError("RPI factor does not reconcile to its declared indices")
    return {
        "base_obligation_year": base["obligation_year"],
        "base_buyout_price_gbp_per_roc": float(base["buyout_price_gbp_per_roc"]),
        "base_recycle_value_gbp_per_roc": float(base["final_recycle_value_gbp_per_roc"]),
        "base_notional_roc_value_gbp_per_roc": float(base["final_notional_roc_value_gbp_per_roc"]),
        "rpi_anchor_year": int(base["rpi_anchor_year"]),
        "rpi_restatement_factor": factor,
        "buyout_price_gbp_per_roc_report_prices": float(base["buyout_price_gbp_per_roc"]) * factor,
        "recycle_value_gbp_per_roc_report_prices": float(base["final_recycle_value_gbp_per_roc"]) * factor,
        "notional_roc_value_gbp_per_roc_report_prices": (
            float(base["final_notional_roc_value_gbp_per_roc"]) * factor),
        "report_price_year": int(rpi["to_price_year"]),
        "method": (
            "The RO buy-out price is RPI-indexed each obligation year, so it is flat in real RPI "
            "terms. The last sourced notional value (buy-out plus final recycle) is restated at "
            "the report price year using the ONS RPI all-items annual index. No RPI forecast is "
            "used and no obligation year beyond the sourced one is projected."
        ),
        "limitations": [
            "The report basis is real CPIH while the buy-out is RPI-indexed. RPI has run above "
            "CPIH, so holding the buy-out flat in real RPI terms lets it drift slightly upward in "
            f"real CPIH terms by {int(model_year)}. The drift is stated rather than modelled.",
            "The recycle value depends on how far the obligation exceeds certificate supply in "
            "the year. It is carried at its sourced ratio to the buy-out price; a larger "
            f"shortfall in {fiscal_year_label(model_year, '/')} would raise it.",
        ],
        "source": base["source_url"],
    }


def closed_scheme_cost(register_path, parameters: dict, recovery: dict, year: int) -> tuple[pd.DataFrame, dict]:
    """The RO in a bill year after the scheme closed: a declared nil (task Y13, the 2050 bill).

    The RO closes to all generation support on 31 March 2037 (:data:`RO_SCHEME_END`), so
    no accreditation runs in the year and nothing is read from the networks. The scheme
    parameters must say so (``scheme_closed``), so a nil is never produced by accident.

    Args:
        register_path: the Ofgem accredited-stations register (hashed as the evidence).
        parameters: the year's RO scheme parameters, with ``scheme_closed`` and ``relief_cases``.
        recovery: the recovery-demand file (its ``forecast_year`` is ``year``).
        year: the bill's calendar year.

    Returns:
        An empty per-carrier frame and the summary the bill reads, every cost zero.
    """
    closed = parameters.get("scheme_closed") or {}
    if not closed.get("reason") or not closed.get("closed_on"):
        raise ValueError(
            f"calendar {year} starts after the RO closed ({RO_SCHEME_END.date()}); the scheme parameters must "
            "declare scheme_closed with its reason and closed_on date")
    if pd.Timestamp(closed["closed_on"]) != RO_SCHEME_END:
        raise ValueError(f"scheme_closed.closed_on {closed['closed_on']} is not the scheme end {RO_SCHEME_END.date()}")
    fleet = read_eligible_fleet(register_path, obligation_year_start=pd.Timestamp(f"{int(year)}-04-01"),
                                period=calendar_period(year))
    if bool(fleet.eligible.any()):
        raise ValueError("a station still holds an accreditation after the scheme end; check the register")
    label, start, end, _ = bill_period(year)
    columns = ["carrier", "stations", "eligible_capacity_mw", "applied_load_factor", "estimated_generation_mwh",
               "roc_banding", "roc_volume"]
    summary = {
        "purpose": f"Renewables Obligation in {label}: the scheme closed on {RO_SCHEME_END.date()}, so the line is nil.",
        "method_label": "scheme_closed",
        "obligation_year": label,
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat(),
                   "basis": f"scheme closed before calendar {int(year)}"},
        "scheme_closed": closed,
        "eligible_capacity_mw": 0.0,
        "expired_capacity_mw": float(fleet["capacity_mw"].sum()),
        "register_total_capacity_mw": float(fleet["capacity_mw"].sum()),
        "eligible_stations": 0, "estimated_generation_mwh": 0.0,
        "roc_volume": 0.0, "uk_roc_volume": 0.0,
        "notional_roc_value": {"report_price_year": int(parameters["report_price_year"]),
                               "notional_roc_value_gbp_per_roc_report_prices": 0.0,
                               "note": "no obligation, so no certificate value"},
        "gross_scheme_cost_gbp": 0.0,
        "obligation_eligible_supply_mwh": float(recovery["ro_obligation_eligible_supply_mwh"]["value"]),
        "gross_gbp_per_3100kwh_ex_vat": 0.0,
        "relief_cases": {name: {"exchequer_relief_share": float(share), "gbp_per_3100kwh_ex_vat": 0.0}
                         for name, share in parameters["relief_cases"].items()},
        "reference_external_gbp_per_3100kwh_ex_vat": float(parameters["reference_external_gbp_per_3100kwh_ex_vat"]),
        "limitations": list(parameters.get("limitations") or []),
        "sources": {
            "ofgem_ro_register": {
                "local_file": Path(register_path).as_posix(),
                "sha256": hashlib.sha256(Path(register_path).read_bytes()).hexdigest(),
                "accreditation_rule": (f"{ROC_ACCREDITATION_YEARS} years from the register effective date, capped at "
                                       f"the scheme end {RO_SCHEME_END.date()}: no station runs in calendar {int(year)}"),
            },
            "scheme_parameters": parameters,
        },
    }
    return pd.DataFrame(columns=columns), summary


def build_ro_scheme_cost(register_path, networks, parameters, recovery, rpi,
                         bandings: dict, certificates_path=None) -> tuple[pd.DataFrame, dict]:
    """Return per-carrier ROC volume and the gross scheme cost per customer.

    With ``certificates_path`` and ``parameters['certificate_evidence']`` (item
    MR8), each station takes its observed banding where Ofgem has issued it
    certificates, and stations of the configured carriers count only if they
    were issued certificates in or after the configured compliance year. A
    fuelled accreditation earns ROCs only while it burns eligible fuel, so one
    with no recent certificates is not an operating ROC generator (Fawley, an
    oil station closed in 2013, carried 1,000 MW of a 2011 accreditation).
    """
    # Item Y4: the obligation period is the recovery-demand file's forecast year.
    year = recovery_model_year(recovery)
    obligation_label, period_start, period_end, obligation_start = bill_period(year)
    # Task Y13: a bill year that starts after the scheme closed carries a declared nil.
    if period_start > RO_SCHEME_END:
        return closed_scheme_cost(register_path, parameters, recovery, year)
    fleet = read_eligible_fleet(register_path, obligation_year_start=obligation_start,
                                period=(period_start, period_end))
    unmapped = sorted(set(fleet.loc[fleet.carrier.isna(), "Technology"].unique()))
    eligible = fleet[fleet.eligible & fleet.carrier.notna()].copy()
    # Calendar 2030: each station counts for the share of the year it is accredited,
    # so its capacity below is period-weighted (MW over the year).
    eligible["station_capacity_mw"] = eligible["capacity_mw"]
    eligible["capacity_mw"] = eligible["capacity_mw"] * eligible["accredited_share"]

    evidence_cfg = parameters.get("certificate_evidence") or {}
    inactive_rows = pd.DataFrame()
    if evidence_cfg and certificates_path is None:
        raise ValueError("certificate_evidence is configured but no certificates file was given")
    if evidence_cfg:
        evidence = read_certificate_evidence(
            certificates_path, tuple(evidence_cfg.get("schemes", ("RO", "ROS"))))
        eligible = eligible.join(evidence, on="Accreditation reference")
        activity = evidence_cfg.get("activity_rule") or {}
        if activity:
            carriers_checked = set(activity["carriers"])
            since = int(activity["min_compliance_year_start"])
            inactive = eligible["carrier"].isin(carriers_checked) & ~(
                eligible["last_certificate_year"].fillna(0) >= since)
            inactive_rows = eligible[inactive]
            eligible = eligible[~inactive]
        if evidence_cfg.get("observed_banding", False):
            observed = pd.to_numeric(eligible["observed_banding"], errors="coerce")
            default = eligible["carrier"].map(bandings)
            eligible["station_banding"] = observed.where(observed > 0, default)
            eligible["banding_source"] = np.where(
                observed > 0, "observed_certificates", "carrier_default")
    if "station_banding" not in eligible.columns:
        eligible["station_banding"] = eligible["carrier"].map(bandings)
        eligible["banding_source"] = "carrier_default"

    carriers = sorted(eligible.carrier.unique())
    load_factors = carrier_load_factors(networks, carriers).set_index("carrier")
    basis = {**DEFAULT_LOAD_FACTOR_BASIS, **(parameters.get("load_factor_basis") or {})}
    if basis["mode"] not in LOAD_FACTOR_BASIS_MODES:
        raise ValueError(f"load_factor_basis.mode must be one of {LOAD_FACTOR_BASIS_MODES}, not {basis['mode']!r}")
    min_share = float(basis["min_labelled_share_of_register"])
    labelled = (carrier_load_factors(networks, carriers, support_type="ROC").set_index("carrier")
                if basis["mode"] == "roc_labelled_units" else None)
    overrides = parameters.get("load_factor_overrides", {}) or {}
    unknown = sorted(set(overrides).difference(carriers))
    if unknown:
        raise ValueError(f"load_factor_overrides name carriers with no eligible capacity: {unknown}")

    rows, overrides_applied = [], []
    for carrier, group in eligible.groupby("carrier"):
        capacity = float(group.capacity_mw.sum())
        carrier_average = load_factors.at[carrier, "load_factor"]
        carrier_available = carrier_average is not None and np.isfinite(carrier_average)
        labelled_lf, labelled_mw = None, 0.0
        if labelled is not None:
            labelled_mw = float(labelled.at[carrier, "capacity_mw"] or 0.0)
            value = labelled.at[carrier, "load_factor"]
            labelled_lf = float(value) if value is not None and np.isfinite(value) else None
        # The units that actually hold ROCs, where the model holds enough of them;
        # otherwise the carrier average, as before.
        use_labelled = (labelled_lf is not None and capacity > 0
                        and labelled_mw >= min_share * capacity)
        model_load_factor = labelled_lf if use_labelled else carrier_average
        model_available = use_labelled or carrier_available
        model_source = ("pypsa_balancing_dispatch_roc_units" if use_labelled
                        else "pypsa_balancing_dispatch")
        banding = bandings.get(carrier)
        if banding is None:
            raise ValueError(f"No ROC banding declared for carrier {carrier!r}")
        override = overrides.get(carrier)
        if override is not None:
            applied = float(override["value"])
            if not 0.0 < applied <= 1.0:
                raise ValueError(f"load_factor_overrides[{carrier!r}] must lie in (0, 1]")
            source = f"override:{override['source']}"
            overrides_applied.append({
                "carrier": carrier,
                "pypsa_load_factor": float(model_load_factor) if model_available else None,
                "applied_load_factor": applied,
                "source": override["source"],
                "reason": override.get("reason", ""),
            })
        else:
            if not model_available:
                raise ValueError(
                    f"Carrier {carrier!r} has {capacity:,.0f} MW of ROC-eligible capacity but no "
                    "modelled dispatch to take a load factor from"
                )
            applied = float(model_load_factor)
            source = model_source
        energy = capacity * applied * 8760.0
        # Capacity-weighted banding of the carrier's stations: observed per station
        # where certificates exist, the carrier default otherwise.
        effective_banding = (
            float((group.capacity_mw * group.station_banding).sum() / capacity)
            if capacity else float(banding))
        rows.append({
            "carrier": carrier,
            "stations": int(len(group)),
            "eligible_capacity_mw": capacity,
            # The model value is always carried for transparency; the applied
            # value is what the ROC volume is actually built on.
            "pypsa_load_factor": float(model_load_factor) if model_available else np.nan,
            "carrier_average_load_factor": float(carrier_average) if carrier_available else np.nan,
            "roc_labelled_load_factor": labelled_lf if labelled_lf is not None else np.nan,
            "roc_labelled_capacity_mw": labelled_mw,
            "applied_load_factor": applied,
            "load_factor_source": source,
            "estimated_generation_mwh": energy,
            "roc_banding": effective_banding,
            "carrier_default_banding": float(banding),
            "observed_banding_capacity_mw": float(
                group.loc[group.banding_source.eq("observed_certificates"), "capacity_mw"].sum()),
            "roc_volume": energy * effective_banding,
        })
    by_carrier = pd.DataFrame(rows).sort_values("roc_volume", ascending=False)

    # UK certificate volume (GB and NI stations) and the share GB suppliers carry.
    by_jurisdiction = {}
    for jurisdiction, group in eligible.groupby("jurisdiction"):
        rocs = 0.0
        for carrier, sub in group.groupby("carrier"):
            row = by_carrier.set_index("carrier").loc[carrier]
            rocs += float((sub.capacity_mw * sub.station_banding).sum()) * float(row.applied_load_factor) * 8760.0
        by_jurisdiction[jurisdiction] = {"capacity_mw": float(group.capacity_mw.sum()), "roc_volume": rocs}
    split = gb_share_of_uk_obligation(parameters)
    value = notional_roc_value(parameters, rpi, year)
    uk_roc_volume = float(by_carrier.roc_volume.sum())
    roc_volume = uk_roc_volume * split["gb_share_of_uk_obligation"]
    gross_cost = roc_volume * value["notional_roc_value_gbp_per_roc_report_prices"]
    denominator = float(recovery["ro_obligation_eligible_supply_mwh"]["value"])
    if not denominator > 0:
        raise ValueError("RO needs a positive obligation-eligible supply denominator")

    mwh = 3.1
    gross_per_customer = gross_cost / denominator * mwh
    relief_cases = {
        name: {"exchequer_relief_share": float(share),
               "gbp_per_3100kwh_ex_vat": gross_per_customer * (1.0 - float(share))}
        for name, share in parameters["relief_cases"].items()
    }

    summary = {
        "purpose": (
            "Model-informed Renewables Obligation scheme cost for obligation year "
            f"{obligation_label}, replacing an external allowance that hides the end of the "
            "domestic Exchequer relief."
        ),
        "method_label": "pypsa_informed_fleet_register",
        "obligation_year": obligation_label,
        "period": {"start": period_start.date().isoformat(), "end": period_end.date().isoformat(),
                   "basis": f"accreditation-weighted over calendar {year}"},
        # Period-weighted: a station accredited for part of 2030 counts for that part.
        "eligible_capacity_mw": float(by_carrier.eligible_capacity_mw.sum()),
        "expired_capacity_mw": float(
            fleet.loc[~fleet.eligible, "capacity_mw"].sum()),
        # The part of the year a partly accredited station is not accredited.
        "unaccredited_part_of_year_capacity_mw": float(
            (fleet.loc[fleet.eligible & fleet.carrier.notna(), "capacity_mw"]
             * (1.0 - fleet.loc[fleet.eligible & fleet.carrier.notna(), "accredited_share"])).sum()),
        # Still accredited but excluded by the certificate activity rule (item MR8):
        # eligible + part-year + expired + inactive + unmapped = the register total.
        "inactive_capacity_mw": float(inactive_rows["capacity_mw"].sum()) if len(inactive_rows) else 0.0,
        "register_total_capacity_mw": float(fleet.capacity_mw.sum()),
        "load_factor_basis": basis,
        "eligible_stations": int(by_carrier.stations.sum()),
        "estimated_generation_mwh": float(by_carrier.estimated_generation_mwh.sum()),
        # GB suppliers' share of the UK certificate volume (the bill's quantity).
        "roc_volume": roc_volume,
        "uk_roc_volume": uk_roc_volume,
        "roc_volume_by_jurisdiction": by_jurisdiction,
        "uk_obligation_split": split,
        "gross_gbp_per_3100kwh_if_ni_stations_excluded": (
            by_jurisdiction.get("GB", {}).get("roc_volume", 0.0)
            * value["notional_roc_value_gbp_per_roc_report_prices"] / denominator * 3.1),
        "notional_roc_value": value,
        "gross_scheme_cost_gbp": gross_cost,
        "obligation_eligible_supply_mwh": denominator,
        "gross_gbp_per_3100kwh_ex_vat": gross_per_customer,
        "relief_cases": relief_cases,
        "reference_external_gbp_per_3100kwh_ex_vat": float(
            parameters["reference_external_gbp_per_3100kwh_ex_vat"]),
        "unmapped_register_technologies": unmapped,
        "roc_bandings": bandings,
        "certificate_evidence": {
            "applied": bool(evidence_cfg),
            "settings": evidence_cfg,
            "inactive_capacity_mw_by_carrier": (
                inactive_rows.groupby("carrier")["capacity_mw"].sum().round(3).to_dict()
                if len(inactive_rows) else {}),
            "inactive_stations_largest": [
                {"station": r["Station name"], "technology": r["Technology"],
                 "capacity_mw": round(float(r["capacity_mw"]), 3),
                 "last_certificate_year": (None if pd.isna(r["last_certificate_year"])
                                           else int(r["last_certificate_year"]))}
                for _, r in inactive_rows.sort_values("capacity_mw", ascending=False).head(15).iterrows()
            ] if len(inactive_rows) else [],
            "certificates_file": Path(certificates_path).as_posix() if certificates_path else None,
        },
        "load_factor_overrides_applied": overrides_applied,
        "limitations": [
            "The accredited capacity comes from the Ofgem register, not from the model's ROC "
            "labels: the brownfield networks label ROC units from REPD, which carry no "
            "accreditation dates, so the register decides which stations still earn ROCs in "
            f"calendar {year}, and for what share of it. The model supplies the load factor.",
            "Where the model holds enough ROC-labelled capacity (load_factor_basis) the load "
            "factor is those units' own balancing dispatch, which reflects their sites and their "
            f"negative (minus ROC value) bids; elsewhere it is the carrier average of the {year} "
            "fleet, which runs above an older legacy fleet. The labelled units are the REPD "
            "operating fleet, not the register stations one for one, so their average stands in "
            "for the register fleet's (roc_labelled_capacity_mw against eligible_capacity_mw).",
            "Load factors come from the balancing dispatch, which is the metered basis ROCs are "
            "issued on. Dispatchable carriers are sensitive to it: their physical load factor "
            "depends on how often the wholesale merit order calls them and how often balancing "
            "then uses them to resolve constraints, so their ROC volume reflects the model's "
            "merit order rather than observed issuance. Where a carrier's model load factor is "
            "not a credible issuance basis, load_factor_overrides in the scheme parameters "
            "replaces it and the substitution is recorded per carrier "
            "(load_factor_source, load_factor_overrides_applied); the model value is still "
            "carried as pypsa_load_factor.",
            "ROC bandings are observed per station from the Ofgem certificate register where the "
            "station has been issued certificates (certificate_evidence); stations with none use "
            "the final-accreditation-period carrier default.",
            "Obligation-eligible supply is taken as the CfD eligible-supply volume. Both schemes "
            "exempt energy-intensive industry, but their exemption scopes are defined separately "
            "and are not identical.",
            "Certificate supply is estimated from capacity and load factor, not from the "
            "certificates Ofgem actually issues. It is not a settlement reconciliation.",
            "Northern Ireland (NIRO) stations are counted in the UK certificate volume because "
            "NIROCs are presented against GB obligations in the UK-wide ROC market and DESNZ sets "
            "the UK obligation from UK-wide issuance; GB suppliers carry the GB share of the UK "
            f"obligation (uk_obligation_split, DESNZ 2026/27 calculation held for {fiscal_year_label(year, '/')}). NI "
            "stations take the GB model's load factors, since the model has no NI network.",
        ],
        "sources": {
            "ofgem_ro_register": {
                "local_file": Path(register_path).as_posix(),
                "sha256": hashlib.sha256(Path(register_path).read_bytes()).hexdigest(),
                "accreditation_rule": (
                    f"{ROC_ACCREDITATION_YEARS} years from the register effective date, capped at "
                    f"the scheme end {RO_SCHEME_END.date()}; weighted by the share of "
                    f"{period_start.date()} to {period_end.date()} it is still running"
                ),
            },
            "rpi": rpi,
            "scheme_parameters": parameters,
        },
    }
    return by_carrier, summary


def write_ro_scheme_cost(register_path, networks, parameters_path, recovery_path, output_csv,
                         output_summary, certificates_path=None):
    parameters = json.loads(Path(parameters_path).read_text(encoding="utf-8"))
    recovery = json.loads(Path(recovery_path).read_text(encoding="utf-8"))
    # A closed-scheme year (task Y13) declares no RPI restatement or bandings; every open year does.
    closed = "scheme_closed" in parameters
    by_carrier, summary = build_ro_scheme_cost(
        register_path, networks, parameters, recovery,
        parameters.get("rpi_restatement") if closed else parameters["rpi_restatement"],
        parameters.get("roc_bandings") if closed else parameters["roc_bandings"],
        certificates_path=certificates_path,
    )
    for path in (output_csv, output_summary):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    by_carrier.to_csv(output_csv, index=False)
    Path(output_summary).write_text(json.dumps(summary, indent=2, allow_nan=False),
                                    encoding="utf-8")
    return summary


if "snakemake" in globals():
    write_ro_scheme_cost(
        snakemake.input.register, list(snakemake.input.networks),
        snakemake.input.parameters, snakemake.input.recovery_demand,
        snakemake.output.by_carrier, snakemake.output.summary,
        certificates_path=getattr(snakemake.input, "certificates", None) or None,
    )
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", required=True)
    parser.add_argument("--network", action="append", required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--recovery-demand", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--certificates", default=None)
    args = parser.parse_args()
    result = write_ro_scheme_cost(args.register, args.network, args.parameters,
                                  args.recovery_demand, args.output_csv, args.output_summary,
                                  certificates_path=args.certificates)
    print(f"RO {result['obligation_year']}: {result['eligible_capacity_mw']:,.0f} MW eligible, "
          f"{result['roc_volume'] / 1e6:,.1f}m ROCs, "
          f"gross GBP{result['gross_gbp_per_3100kwh_ex_vat']:,.2f}/customer")
    for name, case in result["relief_cases"].items():
        print(f"  {name}: GBP{case['gbp_per_3100kwh_ex_vat']:,.2f} "
              f"(relief {case['exchequer_relief_share']:.0%})")
