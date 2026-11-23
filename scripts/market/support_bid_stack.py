"""Calibrate CfD and ROC BM prices from historical ELEXON BOD panels."""

from pathlib import Path
from typing import Iterable
import logging
import numpy as np
import pandas as pd
import pypsa

COLUMNS = [
    "carrier", "support_type", "n_bmus", "n_offer_bmus", "n_bid_bmus",
    "offer_price_low", "offer_price_central", "offer_price_high",
    "bid_price_low", "bid_price_central", "bid_price_high",
]


def _quote(panel, bmu, max_abs_price):
    if bmu not in panel:
        return np.nan
    value = pd.to_numeric(panel[bmu], errors="coerce")
    value = value[(value != 0.0) & (value.abs() <= max_abs_price)].dropna()
    return float(value.median()) if not value.empty else np.nan


def _source_records(network_path, offers_path, bids_path, mapping_path,
                    max_abs_price, methods):
    network = pypsa.Network(str(network_path))
    offers, bids, mapping = pd.read_csv(offers_path), pd.read_csv(bids_path), pd.read_csv(mapping_path)
    needed = {"bmu_id", "generator_name", "carrier", "match_method"}
    if missing := needed - set(mapping.columns):
        raise ValueError(f"BMU mapping is missing columns: {sorted(missing)}")
    if "support_type" not in network.generators.columns:
        return pd.DataFrame()
    mapping = mapping[mapping.match_method.astype(str).isin(methods)]
    rows = []
    for item in mapping.dropna(subset=["bmu_id", "generator_name", "carrier"]).drop_duplicates(
        ["bmu_id", "generator_name"]
    ).itertuples(index=False):
        if item.generator_name not in network.generators.index:
            continue
        support = network.generators.at[item.generator_name, "support_type"]
        support = str(support).strip().lower() if pd.notna(support) else "merchant"
        if support not in {"cfd", "roc"}:
            continue
        offer = _quote(offers, str(item.bmu_id), max_abs_price)
        raw_bid = _quote(bids, str(item.bmu_id), max_abs_price)
        if not (np.isfinite(offer) or np.isfinite(raw_bid)):
            continue
        rows.append({
            "bmu_id": str(item.bmu_id), "generator": str(item.generator_name),
            "carrier": str(item.carrier), "support_type": support,
            "match_method": str(item.match_method), "offer_price": offer,
            "bid_price": -raw_bid if np.isfinite(raw_bid) else np.nan,
        })
    return pd.DataFrame(rows)


def calibrate_support_bid_stack(
    calibration_inputs: Iterable[tuple[str | Path, str | Path, str | Path, str | Path]],
    max_abs_price=2000.0, quantiles=(0.25, 0.5, 0.75),
    match_methods=("direct", "power_station_dictionary_direct"),
):
    """Calculate robust BMU-median quote quantiles for each carrier/support pair."""
    frames = [
        _source_records(*item, max_abs_price=max_abs_price, methods=set(match_methods))
        for item in calibration_inputs
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("No CfD/ROC BMUs with usable high-quality ELEXON matches")
    raw = pd.concat(frames, ignore_index=True)
    evidence = raw.groupby(
        ["bmu_id", "generator", "carrier", "support_type", "match_method"],
        as_index=False, dropna=False,
    ).median(numeric_only=True)
    rows = []
    for (carrier, support), group in evidence.groupby(["carrier", "support_type"], sort=True):
        offer, bid = group.offer_price.dropna(), group.bid_price.dropna()
        if offer.empty and bid.empty:
            continue
        rows.append({
            "carrier": carrier, "support_type": support, "n_bmus": int(group.bmu_id.nunique()),
            "n_offer_bmus": int(offer.size), "n_bid_bmus": int(bid.size),
            "offer_price_low": float(offer.quantile(quantiles[0])) if not offer.empty else np.nan,
            "offer_price_central": float(offer.quantile(quantiles[1])) if not offer.empty else np.nan,
            "offer_price_high": float(offer.quantile(quantiles[2])) if not offer.empty else np.nan,
            "bid_price_low": float(bid.quantile(quantiles[0])) if not bid.empty else np.nan,
            "bid_price_central": float(bid.quantile(quantiles[1])) if not bid.empty else np.nan,
            "bid_price_high": float(bid.quantile(quantiles[2])) if not bid.empty else np.nan,
        })
    params = pd.DataFrame(rows, columns=COLUMNS)
    if params.empty:
        raise ValueError("No support groups have both usable offer and bid evidence")
    return params, evidence


def legal_cfd_labels_from_crosswalk(path: str | Path) -> pd.DataFrame:
    """Return dated, high-confidence LCCC CfD BMU labels for historic samples."""
    frame = pd.read_csv(path)
    required = {
        "bmu_id", "support_type", "confidence", "support_conflict",
        "lccc_technology", "cfd_valid_from", "historical_bid_calibration_eligible",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Legal CfD crosswalk is missing columns: {sorted(missing)}")
    technology = frame["lccc_technology"].fillna("").astype(str).str.strip().str.lower()
    carrier = pd.Series(pd.NA, index=frame.index, dtype="object")
    carrier.loc[technology.str.contains("offshore wind", regex=False)] = "wind_offshore"
    carrier.loc[technology.str.contains("onshore wind", regex=False)] = "wind_onshore"
    carrier.loc[technology.str.contains("solar", regex=False)] = "solar_pv"
    valid_from = pd.to_datetime(frame["cfd_valid_from"], errors="coerce")
    eligible = frame["historical_bid_calibration_eligible"].astype(str).str.lower().isin(["true", "1"])
    mask = (
        frame["support_type"].astype(str).str.lower().eq("cfd")
        & frame["confidence"].astype(str).str.lower().eq("high")
        & ~frame["support_conflict"].fillna(False).astype(bool)
        & carrier.notna()
        & eligible
        & valid_from.notna()
    )
    labels = frame.loc[mask, ["bmu_id"]].copy()
    labels["carrier"] = carrier.loc[labels.index]
    labels["support_type"] = "cfd"
    labels["match_method"] = "legal_lccc_cfd_dated"
    labels["cfd_valid_from"] = valid_from.loc[labels.index]
    return labels.drop_duplicates(["bmu_id", "carrier", "support_type"])


def augment_bids_from_renewable_cache(parameters, seed_evidence, cache_dir, years,
                                      max_abs_price=2000.0, quantiles=(0.25, 0.5, 0.75),
                                      additional_labels: pd.DataFrame | None = None):
    """Calibrate bid prices from one time-valid, multi-year value per BMU."""
    labels = seed_evidence[["bmu_id", "carrier", "support_type", "match_method"]].drop_duplicates().copy()
    labels["cfd_valid_from"] = pd.NaT
    if additional_labels is not None and not additional_labels.empty:
        labels = pd.concat([labels, additional_labels], ignore_index=True, sort=False)
        labels = labels.drop_duplicates(["bmu_id", "carrier", "support_type"], keep="last")
    labels["cache_bmu_id"] = labels["bmu_id"].astype(str).str.replace(r"^[ET]_", "", regex=True)
    labels["cfd_valid_from"] = pd.to_datetime(labels["cfd_valid_from"], errors="coerce")
    if labels.empty:
        return parameters, pd.DataFrame()
    cache_dir = Path(cache_dir)
    label_dates = labels.set_index("cache_bmu_id")["cfd_valid_from"]
    pieces = []
    for year in [int(year) for year in years]:
        path = cache_dir / str(year) / "renewable_bids_cache.csv"
        if not path.exists():
            raise FileNotFoundError(f"Renewable ELEXON bid cache not found: {path}")
        year_rows = []
        for chunk in pd.read_csv(path, usecols=["bmu_id", "datetime", "bid_price"], chunksize=500_000):
            chunk["cache_bmu_id"] = chunk["bmu_id"].astype(str).str.replace(r"^[ET]_", "", regex=True)
            chunk = chunk[chunk["cache_bmu_id"].isin(label_dates.index)]
            if chunk.empty:
                continue
            observed_at = pd.to_datetime(chunk["datetime"], errors="coerce")
            valid_from = chunk["cache_bmu_id"].map(label_dates)
            chunk = chunk[valid_from.isna() | observed_at.ge(valid_from)].copy()
            if chunk.empty:
                continue
            values = pd.to_numeric(chunk["bid_price"], errors="coerce")
            chunk = chunk[(values != 0.0) & (values.abs() <= max_abs_price)].copy()
            chunk["bid_price"] = -pd.to_numeric(chunk["bid_price"], errors="coerce")
            year_rows.append(chunk)
        if not year_rows:
            continue
        frame = pd.concat(year_rows, ignore_index=True)
        frame = frame.groupby("cache_bmu_id", as_index=False)["bid_price"].median()
        frame["year"] = year
        pieces.append(frame)
    if not pieces:
        return parameters, pd.DataFrame()

    annual_bmu_values = pd.concat(pieces, ignore_index=True)
    cache_evidence = annual_bmu_values.groupby("cache_bmu_id", as_index=False).agg(
        bid_price=("bid_price", "median"), n_cache_years=("year", "nunique"),
    ).merge(labels, on="cache_bmu_id", how="inner")
    params = parameters.copy().set_index(["carrier", "support_type"], drop=False)
    for key, group in cache_evidence.groupby(["carrier", "support_type"], sort=True):
        bid = group["bid_price"].dropna()
        if bid.empty:
            continue
        values = {
            "bid_price_low": float(bid.quantile(quantiles[0])),
            "bid_price_central": float(bid.quantile(quantiles[1])),
            "bid_price_high": float(bid.quantile(quantiles[2])),
            "n_bid_bmus": int(group["bmu_id"].nunique()),
        }
        if key not in params.index:
            params.loc[key, "carrier"] = key[0]
            params.loc[key, "support_type"] = key[1]
            params.loc[key, "n_bmus"] = int(group["bmu_id"].nunique())
            params.loc[key, "n_offer_bmus"] = 0
            params.loc[key, ["offer_price_low", "offer_price_central", "offer_price_high"]] = np.nan
        for column, value in values.items():
            params.loc[key, column] = value
    return params.reset_index(drop=True)[COLUMNS], cache_evidence

def augment_offers_from_renewable_cache(parameters, seed_evidence, cache_dir, years,
                                        max_abs_price=2000.0, quantiles=(0.25, 0.5, 0.75)):
    """Replace offer assumptions with multi-year cached renewable BOD offers."""
    labels = seed_evidence[["bmu_id", "carrier", "support_type", "match_method"]].drop_duplicates().copy()
    labels["cache_bmu_id"] = labels["bmu_id"].astype(str).str.replace(r"^[ET]_", "", regex=True)
    cache_dir = Path(cache_dir)
    pieces = []
    for year in [int(year) for year in years]:
        path = cache_dir / str(year) / "renewable_offers_cache.csv"
        if not path.exists():
            raise FileNotFoundError(f"Renewable ELEXON offer cache not found: {path}")
        year_rows = []
        usecols = ["bmu_id", "datetime", "settlement_period", "pair_id", "level_from", "level_to", "offer_price"]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
            chunk["cache_bmu_id"] = chunk["bmu_id"].astype(str).str.replace(r"^[ET]_", "", regex=True)
            chunk = chunk[chunk["cache_bmu_id"].isin(set(labels["cache_bmu_id"]))]
            if chunk.empty:
                continue
            values = pd.to_numeric(chunk["offer_price"], errors="coerce")
            pair = pd.to_numeric(chunk["pair_id"], errors="coerce")
            chunk = chunk[(pair > 0) & (values != 0.0) & (values.abs() <= max_abs_price)].copy()
            year_rows.append(chunk)
        if not year_rows:
            continue
        frame = pd.concat(year_rows, ignore_index=True)
        keys = ["cache_bmu_id", "datetime", "settlement_period"]
        frame["_level"] = pd.concat([
            pd.to_numeric(frame["level_from"], errors="coerce").abs(),
            pd.to_numeric(frame["level_to"], errors="coerce").abs(),
        ], axis=1).max(axis=1).fillna(0.0)
        frame = frame.sort_values(keys + ["pair_id"], kind="mergesort")
        cumulative = frame.groupby(keys, sort=False)["_level"].cummax()
        previous = cumulative.groupby([frame[key] for key in keys], sort=False).shift(fill_value=0.0)
        frame["_volume"] = (frame["_level"] - previous).clip(lower=0.0)
        frame["_volume"] = frame["_volume"].where(frame["_volume"] > 0, frame["_level"])
        frame["_weighted_price"] = frame["offer_price"] * frame["_volume"]
        weighted = frame.groupby(keys, sort=False).agg(
            _weighted_price=("_weighted_price", "sum"),
            _volume=("_volume", "sum"),
            _offer_mean=("offer_price", "mean"),
        ).reset_index()
        weighted["offer_price"] = np.where(
            weighted["_volume"] > 0,
            weighted["_weighted_price"] / weighted["_volume"],
            weighted["_offer_mean"],
        )
        frame = weighted.groupby("cache_bmu_id", as_index=False)["offer_price"].median()
        frame["year"] = year
        pieces.append(frame)
    if not pieces:
        return parameters, pd.DataFrame()

    cache_evidence = pd.concat(pieces, ignore_index=True).merge(labels, on="cache_bmu_id", how="inner")
    params = parameters.copy().set_index(["carrier", "support_type"], drop=False)
    for key, group in cache_evidence.groupby(["carrier", "support_type"], sort=True):
        offer = group["offer_price"].dropna()
        if offer.empty:
            continue
        values = {
            "offer_price_low": float(offer.quantile(quantiles[0])),
            "offer_price_central": float(offer.quantile(quantiles[1])),
            "offer_price_high": float(offer.quantile(quantiles[2])),
            "n_offer_bmus": int(group["bmu_id"].nunique()),
        }
        if key not in params.index:
            params.loc[key, "carrier"] = key[0]
            params.loc[key, "support_type"] = key[1]
            params.loc[key, "n_bmus"] = int(group["bmu_id"].nunique())
            params.loc[key, "n_bid_bmus"] = 0
            params.loc[key, ["bid_price_low", "bid_price_central", "bid_price_high"]] = np.nan
        for column, value in values.items():
            params.loc[key, column] = value
    return params.reset_index(drop=True)[COLUMNS], cache_evidence


def apply_empirical_support_prices(network, bm_bids, gen_offer, gen_bid, logger: logging.Logger):
    """Use historical support quotes only where there are enough direct BMU matches.

    ``bm_bids`` is the scenario's ``support.bm_bids`` block.
    """
    config = bm_bids or {}
    case = str(config.get("case", "central")).lower()
    if case not in {"low", "central", "high"}:
        raise ValueError("support.bm_bids.case must be low, central, or high")
    path = Path(config.get("parameters_file", ""))
    if not path.exists():
        raise FileNotFoundError(f"Empirical support pricing requires parameters_file: {path}")
    parameters = pd.read_csv(path).set_index(["carrier", "support_type"])
    required = {f"offer_price_{case}", f"bid_price_{case}", "n_offer_bmus", "n_bid_bmus"}
    if missing := required - set(parameters.columns):
        raise ValueError(f"Support bid-stack parameter file missing columns: {sorted(missing)}")
    minimum = int(config.get("min_bmus", 3))
    gen_offer, gen_bid = gen_offer.copy(), gen_bid.copy()
    applied, fallback = [], []
    if "support_type" not in network.generators.columns:
        logger.warning("Empirical support pricing enabled but support_type labels are absent")
        return gen_offer, gen_bid
    for name, generator in network.generators.iterrows():
        support = generator.get("support_type", "merchant")
        support = str(support).strip().lower() if pd.notna(support) else "merchant"
        if support not in {"cfd", "roc"}:
            continue
        key = (str(generator.get("carrier", "unknown")), support)
        if key not in parameters.index:
            fallback.append("/".join(key))
            continue
        row = parameters.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        applied_sides = []
        offer = pd.to_numeric(pd.Series([row[f"offer_price_{case}"]]), errors="coerce").iloc[0]
        bid = pd.to_numeric(pd.Series([row[f"bid_price_{case}"]]), errors="coerce").iloc[0]
        if int(row.n_offer_bmus) >= minimum and np.isfinite(offer):
            gen_offer.at[name] = float(offer)
            applied_sides.append("offer")
        if int(row.n_bid_bmus) >= minimum and np.isfinite(bid):
            gen_bid.at[name] = float(bid)
            applied_sides.append("bid")
        if applied_sides:
            applied.append("/".join(key) + " (" + ", ".join(applied_sides) + ")")
        else:
            fallback.append("/".join(key))
    logger.info("Applied empirical support-type BM prices to %d generators: %s",
                len(applied), ", ".join(sorted(set(applied))) or "none")
    if fallback:
        logger.info("Retained configured support prices where direct historic evidence is insufficient: %s",
                    ", ".join(sorted(set(fallback))))
    return gen_offer, gen_bid


def main(snakemake):
    from scripts.utilities.logging_config import setup_logging
    logger = setup_logging(snakemake.log[0])
    cfg = snakemake.params.support_bid_stack
    parameters, evidence = calibrate_support_bid_stack(
        list(zip(snakemake.input.calibration_networks, snakemake.input.calibration_offers,
                 snakemake.input.calibration_bids, snakemake.input.calibration_mappings)),
        max_abs_price=float(cfg.get("max_abs_historic_price", 2000.0)),
        quantiles=tuple(cfg.get("quantiles", [0.25, 0.5, 0.75])),
        match_methods=cfg.get("mapping_match_methods",
                              ["direct", "power_station_dictionary_direct"]),
    )
    legal_labels = pd.DataFrame(columns=["bmu_id", "carrier", "support_type", "match_method"])
    legal_cfg = cfg.get("legal_cfd_crosswalk", {}) or {}
    if legal_cfg.get("enabled", False):
        legal_path = Path(legal_cfg.get("file", ""))
        if not legal_path.exists():
            raise FileNotFoundError(f"Legal CfD crosswalk not found: {legal_path}")
        legal_labels = legal_cfd_labels_from_crosswalk(legal_path)
        logger.info("Loaded %d high-confidence LCCC CfD BMU labels from %s", len(legal_labels), legal_path)
    cache_years = cfg.get("cached_bid_years", []) or []
    if cache_years:
        parameters, cache_evidence = augment_bids_from_renewable_cache(
            parameters,
            evidence,
            cache_dir=cfg.get("renewable_bid_cache_dir", "resources/market/elexon"),
            years=cache_years,
            max_abs_price=float(cfg.get("max_abs_historic_price", 2000.0)),
            quantiles=tuple(cfg.get("quantiles", [0.25, 0.5, 0.75])),
            additional_labels=legal_labels,
        )
        evidence = pd.concat(
            [evidence.assign(source="scenario_BOD"),
             cache_evidence.assign(source="renewable_bid_cache")],
            ignore_index=True,
            sort=False,
        )
    offer_cache_years = cfg.get("cached_offer_years", []) or []
    if offer_cache_years:
        parameters, offer_cache_evidence = augment_offers_from_renewable_cache(
            parameters,
            evidence[evidence.get("source", "scenario_BOD") == "scenario_BOD"],
            cache_dir=cfg.get("renewable_offer_cache_dir", "resources/market/elexon"),
            years=offer_cache_years,
            max_abs_price=float(cfg.get("max_abs_historic_price", 2000.0)),
            quantiles=tuple(cfg.get("quantiles", [0.25, 0.5, 0.75])),
        )
        evidence = pd.concat(
            [evidence, offer_cache_evidence.assign(source="renewable_offer_cache")],
            ignore_index=True,
            sort=False,
        )
    parameter_path, evidence_path = Path(snakemake.output.parameters_csv), Path(snakemake.output.evidence_csv)
    parameter_path.parent.mkdir(parents=True, exist_ok=True)
    parameters.to_csv(parameter_path, index=False)
    evidence.to_csv(evidence_path, index=False)
    logger.info("Wrote support bid-stack parameters: %s (%d rows)", parameter_path, len(parameters))
    logger.info("Wrote support bid-stack evidence: %s (%d independent BMUs)", evidence_path, len(evidence))


if __name__ == "__main__":
    main(snakemake)
