"""Support labels for named renewable sites from the scheme registers.

REPD is a planning database, not a support register. Its ``RO Banding`` field
is blank for about nine sites in ten and its ``CfD Capacity`` field is blank for
some contracted projects, so labels built from REPD alone go wrong on real
plant. The CP30 2030 build (review, 26 Sep 2026) had:

* Race Bank, Rampion, Galloper and the Aberdeen demonstrator (EOWDC), 1.4 GW,
  labelled merchant. All four were accredited under the RO in the grace period
  after the 1 April 2017 closure, which the date-based inference in
  ``_assign_subsidy_attributes`` cannot see. Onshore was short of the Ofgem
  register for the same reason, while solar was over-labelled: the date rule
  makes every pre-2017 REPD solar farm ROC, including the Feed-in Tariff ones.
* Triton Knoll (857 MW, AR2) and Eastfields (AR4) labelled merchant because REPD
  gives the round but not the contracted capacity, and the REPD branch requires
  a positive ``cfd_capacity_mw``.

This module makes the scheme registers the authority for the named fleet:

* **CfD** (:func:`apply_lccc_cfd_evidence`, before the subsidy assignment). A
  named site matched to a live or pre-start LCCC contract gets the LCCC round and
  contract capacity where REPD gives none, so the REPD branch labels it CfD. A
  REPD round with no REPD capacity counts only when LCCC confirms it. A round
  REPD already records with a capacity is never changed here (sourced
  corrections live in ``data/market/lccc/cfd_round_corrections.csv``); a
  disagreement is reported.
* **RO** (:func:`apply_ro_register_labels`, after it). A named operational site
  matched to an Ofgem accreditation still running on the reference date (20
  years from the effective date, capped at the scheme end on 31 March 2037) is
  ROC with the station's banding, observed from the Ofgem certificate register
  where issued. A site matched only to expired accreditations, or to none, is
  not ROC. The eligibility rule is the one
  ``scripts/consumer_bill/build_ro_eligible_fleet.py`` uses for the bill's RO
  line (imported, not copied), so the labelled fleet and the RO line count the
  same stations.

Precedence follows the codebase rule in ``_assign_subsidy_attributes``: a CfD
site stays CfD even where it also holds an RO accreditation. Proxy (FES
residual) rows are never touched.

Matching is by normalised name (unicode spaces including ``\\xa0``, apostrophes,
punctuation, and naming boilerplate such as "Wind Farm" or "Solar Park"
removed), capacity and, where both records carry one, postcode, with a sourced
override file for pairs the names cannot settle. UK site names repeat ("Manor
Farm", "Low Farm"), so a fuzzy match needs a *distinctive* shared name token or
a matching postcode, and phase or extension markers must agree. Every decision
is written to an audit table.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from scripts.consumer_bill.build_ro_eligible_fleet import (
    read_certificate_evidence,
    read_eligible_fleet,
)

LOGGER = logging.getLogger(__name__)

# Carriers of the renewables stage whose named sites can hold an RO or CfD.
SITE_CARRIERS = ("wind_onshore", "wind_offshore", "solar_pv")

# Register stations below this size are not in REPD (its threshold is 150 kW,
# 1 MW for older records), so they are not searched.
MIN_MATCH_CAPACITY_MW = 0.1

# Capacity agreement: Ofgem records declared net capacity, REPD installed
# capacity, so a few per cent apart is normal.
CAPACITY_RELATIVE_TOLERANCE = 0.20
CAPACITY_ABSOLUTE_TOLERANCE_MW = 1.0

# A site takes several accreditations (phases, units) up to this multiple of
# its REPD capacity; one accreditation larger than its site by more than this
# is not accepted as that site.
SITE_CAPACITY_HEADROOM = 1.25
MAX_UNIT_TO_SITE_RATIO = 2.0

# A matched site is split into a ROC row and a remainder only when the
# unaccredited remainder is larger than both of these.
SPLIT_MIN_REMAINDER_MW = 1.0
SPLIT_MIN_REMAINDER_SHARE = 0.20
# REPD often gives a solar farm's DC peak and Ofgem its AC declared net
# capacity; DC/AC ratios up to about 1.35 are normal, so a solar site is split
# only when the accredited capacity is under 65 per cent of the REPD figure.
SPLIT_MIN_REMAINDER_SHARE_BY_CARRIER = {"solar_pv": 0.35}

POSTCODE_NAME_SCORE = 70.0
TOKEN_FUZZY_SCORE = 88.0
TOKEN_FUZZY_MIN_LENGTH = 6

LCCC_LIVE_STATUSES = ("Live (Post-FIC)", "Live (Pre-FIC)", "Pre-Start Date", "Pre-MDD")

# LCCC technology types that settle at their own strike-table row rather than the
# carrier's (cfd_strike_table.CFD_TECHNOLOGY_COLUMN). The AR6 permitted-reduction
# and AR7 Scottish-pot contracts are not marked in the LCCC file: the checked CfD
# table declares them.
LCCC_STRIKE_TECHNOLOGY = {
    "Floating Offshore Wind": "floating_offshore",
    "Remote Island Wind": "remote_island_wind",
}

LCCC_TECHNOLOGY_TO_CARRIER = {
    "Offshore Wind": "wind_offshore",
    "Floating Offshore Wind": "wind_offshore",
    "Onshore Wind": "wind_onshore",
    "Remote Island Wind": "wind_onshore",
    "Solar PV": "solar_pv",
}

# Calendar year each round was awarded: a site operating before its round was
# awarded cannot be the contracted project (it may be an older neighbour).
LCCC_ROUND_AWARD_YEAR = {
    "Investment Contract": 2014, "AR1": 2015, "AR2": 2017, "AR3": 2019,
    "AR4": 2022, "AR5": 2023, "AR6": 2024, "AR7": 2025, "AR7a": 2025,
}

_SPACE_CHARACTERS = ("\xa0", " ", " ", " ", " ", "﻿")

# Boilerplate removed before names are compared. Order matters: phrases first.
_BOILERPLATE = (
    r"offshore wind ?farm", r"onshore wind ?farm", r"wind ?farm", r"wind ?park",
    r"wind ?turbines?", r"wind ?energy", r"wind ?power", r"wind ?project",
    r"wind ?generating", r"wind ?cluster",
    r"solar ?farm", r"solar ?park", r"solar ?pv", r"solar ?array", r"solar ?project",
    r"solar ?power", r"solar ?energy", r"solar ?panels?", r"photovoltaics?", r"power ?station",
    r"generating ?station", r"renewable ?energy ?(?:project|plant|park|centre)?",
    r"limited", r"ltd", r"llp", r"plc", r"offshore", r"onshore", r"windfarm", r"windpark",
    r"solar", r"pv", r"wind", r"the", r"site", r"project", r"development", r"scheme",
    r"and", r"of", r"at", r"re ?powering", r"repower(?:ed|ing)?", r"full",
)
_BOILERPLATE_RE = re.compile(r"\b(?:" + "|".join(_BOILERPLATE) + r")\b")
_PHASE_RE = re.compile(r"\b(?:phase|stage|ph)\s*(?:[0-9]+|[ivx]+|one|two|three|four|[a-e])\b")
_UNIT_SUFFIX_RE = re.compile(r"\s-\s*[a-e](?:\s*,\s*[a-e])*(?=\s|$|\()", re.IGNORECASE)
_DATE_RE = re.compile(r"\(?\b\d{1,2}/\d{1,2}/\d{2,4}\b\)?")
_POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})\b")

# Words too common in British site names to identify a site by themselves.
GENERIC_NAME_TOKENS = frozenset("""
farm farms manor home grange hall park house lodge field fields barn barns court north south
east west central church mill mills hill hills wood woods green lane road street common cross
moor moss top bottom lower upper great little estate estates airfield extension land lands near
new old high low bank banks end gate bridge brook marsh fen fens heath down downs valley point
dale holdings energy power renewables renewable generation company group limited station
works plant centre center industrial business court place view side rise meadow meadows
leisure school college university hospital council county district garage depot warehouse
unit units quarry reservoir water treatment sewage demonstration demonstrator test ground
way row lea ley oak ash elm bay isle island mount community glebe priory abbey castle
chapel rectory cottage cottages grove orchard croft spring springs pool lake pond forest
airport base raf fruit dairy poultry stud nursery nurseries battery batteries storage bess
infrastructure facility array panels mounted hybrid grid charging hydrogen co operative
resubmission application revised amended variation
""".split())
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4}


def _plain_text(value: object) -> str:
    """Lower-case ASCII text; unicode spaces (``\\xa0``) as spaces, apostrophes dropped."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    for character in _SPACE_CHARACTERS:
        text = text.replace(character, " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"['`]", "", text)
    return text.lower().replace("&", " and ")


def _tokens_text(value: object) -> str:
    text = _DATE_RE.sub(" ", _plain_text(value))
    text = _UNIT_SUFFIX_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    # "Rothes2", "Stroupster1": a number run into a name is a phase number.
    return re.sub(r"(?<=[a-z]{3})(?=\d\b)", " ", text)


def normalise_station_name(value: object) -> str:
    """Name key: boilerplate, punctuation and spacing removed; phase markers kept."""
    return " ".join(_BOILERPLATE_RE.sub(" ", _tokens_text(value)).split())


def phase_free_key(value: object) -> str:
    """:func:`normalise_station_name` without phase or unit markers."""
    text = _PHASE_RE.sub(" ", _tokens_text(value))
    return " ".join(_BOILERPLATE_RE.sub(" ", text).split())


def phase_marker(value: object) -> str:
    """The phase, stage, number or extension marker of a name ('' if none).

    ``Walney 1`` and ``Walney Offshore Wind Phase I`` both give ``1``; ``Kentish
    Flats Extension`` gives ``ext``. Used to keep a fuzzy match from pairing one
    phase of a project with another.
    """
    text = normalise_station_name(value)
    match = re.search(r"\b(?:phase|stage|ph)\s*([0-9]+|[ivx]+|one|two|three|four)\b", text)
    if match:
        token = match.group(1)
    else:
        tokens = text.split()
        if "extension" in tokens or "ext" in tokens:
            return "ext"
        token = tokens[-1] if tokens else ""
        if not (token.isdigit() and len(token) == 1) and token not in ("ii", "iii", "iv"):
            return ""
    if token.isdigit():
        return str(int(token))
    if token in _ROMAN:
        return str(_ROMAN[token])
    if token in _WORD_NUMBERS:
        return str(_WORD_NUMBERS[token])
    return token


def markers_compatible(left: str, right: str) -> bool:
    """Phase markers that can name the same project.

    Equal markers; no marker against phase 1 or against an extension (the
    unnumbered record may be the whole project or its later part, which the
    commissioning-year test then decides); and an extension against phase 2
    or later (``Kentish Flats Extension`` is REPD's ``Kentish Flats 2``).
    """
    if left == right:
        return True
    pair = {left, right}
    if "" in pair:
        return bool(pair & {"1", "ext"})
    if "ext" in pair:
        other = (pair - {"ext"}).pop()
        return other.isdigit() and int(other) >= 2
    return False


def distinctive_tokens(key: str) -> frozenset:
    """Tokens of a name key that can identify a site (length >= 3, not generic)."""
    return frozenset(t for t in key.split() if len(t) >= 3 and not t.isdigit()
                     and t not in GENERIC_NAME_TOKENS)


def _token_in(token: str, tokens: Iterable[str]) -> bool:
    if token in tokens:
        return True
    if len(token) < TOKEN_FUZZY_MIN_LENGTH:
        return False
    return any(len(other) >= TOKEN_FUZZY_MIN_LENGTH and fuzz.ratio(token, other) >= TOKEN_FUZZY_SCORE
               for other in tokens)


def distinctive_overlap(left_key: str, right_key: str, directional: bool = False) -> bool:
    """True when every distinctive token of one name appears in the other.

    ``directional`` requires it of the left name (the register unit): its
    distinctive tokens must all appear in the site name. Project numbers must
    not conflict: ``Hornsea 2`` never matches ``Hornsea 3``, though ``Hornsea``
    matches either.
    """
    left, right = distinctive_tokens(left_key), distinctive_tokens(right_key)
    if not left or not right:
        return False
    left_all, right_all = set(left_key.split()), set(right_key.split())
    left_numbers = {t for t in left_all if t.isdigit()}
    right_numbers = {t for t in right_all if t.isdigit()}
    if left_numbers and right_numbers and not left_numbers & right_numbers:
        return False
    if all(_token_in(t, right_all) for t in left):
        return True
    return not directional and all(_token_in(t, left_all) for t in right)


def postcode(value: object) -> str:
    """The last full UK postcode in a text field, without spaces ('' if none)."""
    matches = _POSTCODE_RE.findall(str(value or "").upper())
    return "".join(matches[-1]) if matches else ""


def capacity_agrees(left: float, right: float,
                    relative: float = CAPACITY_RELATIVE_TOLERANCE,
                    absolute: float = CAPACITY_ABSOLUTE_TOLERANCE_MW) -> bool:
    if pd.isna(left) or pd.isna(right) or left <= 0 or right <= 0:
        return False
    difference = abs(float(left) - float(right))
    return difference <= absolute or difference / max(float(left), float(right)) <= relative


def clean_reference(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "<na>") else re.sub(r"\.0$", "", text)


# ---------------------------------------------------------------------------
# Registers
# ---------------------------------------------------------------------------

def default_banding_for_accreditation(carrier: str, effective_date: pd.Timestamp,
                                      default_bandings: dict) -> float:
    """Banding for a station with no issued certificates to observe it from.

    Offshore and onshore wind follow the bands by accreditation year (offshore
    2.0 to March 2015, 1.9 in 2015/16, 1.8 from April 2016; onshore 1.0 before
    April 2013, 0.9 after). Other carriers take the configured carrier default.
    """
    if pd.notna(effective_date):
        if carrier == "wind_offshore":
            if effective_date < pd.Timestamp("2015-04-01"):
                return 2.0
            if effective_date < pd.Timestamp("2016-04-01"):
                return 1.9
            return 1.8
        if carrier == "wind_onshore":
            return 1.0 if effective_date < pd.Timestamp("2013-04-01") else 0.9
    value = default_bandings.get(carrier)
    return float(value) if value is not None else np.nan


def load_ro_register(register_path, reference_date, certificates_path=None,
                     default_bandings: Optional[dict] = None,
                     schemes: Iterable[str] = ("RO", "ROS")) -> pd.DataFrame:
    """Ofgem RO accredited stations with eligibility and banding on ``reference_date``.

    Eligibility (20 years from the effective date, capped at 31 March 2037) is
    :func:`read_eligible_fleet` from the bill's RO module, so both count the
    same stations. ``banding`` is the observed certificate banding where Ofgem
    has issued certificates, otherwise the accreditation-year default.

    A network label is one date, so the single-date rule (``period=None``) is
    used: a station is eligible if its accreditation still runs on
    ``reference_date``. The bill's RO line weights each station by the share of
    calendar 2030 it holds accreditation instead; since the bill review of
    28 Sep 2026 that is :func:`read_eligible_fleet`'s default, which had leaked
    into these labels and marked as ROC the 40 stations (94 MW) whose
    accreditation ends in Q1 2030 (OW12, restored for the 2030 re-solve).
    """
    frame = read_eligible_fleet(register_path, obligation_year_start=pd.Timestamp(reference_date),
                                period=None)
    frame = frame.rename(columns={"eligible": "ro_eligible"})
    frame["Accreditation reference"] = frame["Accreditation reference"].map(clean_reference)
    country = frame["Country"] if "Country" in frame else pd.Series("", index=frame.index)
    frame["gb"] = country.astype(str).ne("Northern Ireland")
    address = frame["Station address"] if "Station address" in frame else pd.Series("", index=frame.index)
    frame["postcode"] = address.map(postcode)
    if certificates_path:
        evidence = read_certificate_evidence(certificates_path, tuple(schemes))
        frame = frame.join(evidence, on="Accreditation reference")
    else:
        frame["last_certificate_year"] = np.nan
        frame["observed_banding"] = np.nan
    defaults = default_bandings or {}
    fallback = pd.Series([
        default_banding_for_accreditation(carrier, date, defaults)
        for carrier, date in zip(frame["carrier"], frame["effective_date"])
    ], index=frame.index, dtype=float)
    observed = pd.to_numeric(frame["observed_banding"], errors="coerce")
    # A banding outside the RO's range is a certificate artefact, not a band.
    plausible = observed.between(0.25, 5.0)
    frame["banding"] = observed.where(plausible, fallback)
    frame["banding_source"] = np.where(plausible, "observed_certificates", "accreditation_year_default")
    frame["unit_id"] = frame["Accreditation reference"]
    frame["unit_name"] = frame["Station name"]
    frame["unit_capacity_mw"] = frame["capacity_mw"]
    frame["unit_year"] = frame["effective_date"].dt.year
    return frame


def lccc_round_label(value: object) -> str:
    """LCCC 'Allocation Round 2' -> 'AR2'; 'Investment Contract' kept as is."""
    text = " ".join(_plain_text(value).split())
    match = re.fullmatch(r"allocation round (\d+)", text)
    if match:
        return f"AR{match.group(1)}"
    if text == "investment contract":
        return "Investment Contract"
    return str(value).strip()


def load_lccc_portfolio(path) -> pd.DataFrame:
    """LCCC CfD units still holding a contract (not terminated), keyed for matching."""
    frame = pd.read_csv(path)
    required = {"CFD_ID", "Name_of_CFD_Unit", "Allocation_Round", "Technology_Type",
                "Status", "Maximum_Contract_Capacity_MW"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"LCCC CfD portfolio is missing columns: {sorted(missing)}")
    frame = frame[frame["Status"].isin(LCCC_LIVE_STATUSES)].copy()
    frame["carrier"] = frame["Technology_Type"].map(LCCC_TECHNOLOGY_TO_CARRIER)
    frame = frame[frame["carrier"].notna()].copy()
    frame["cfd_round"] = frame["Allocation_Round"].map(lccc_round_label)
    # LCCC files AR7a (onshore, solar) under "Allocation Round 7"; the strike
    # table keys them AR7a.
    from scripts.generators.cfd_vintage_labels import normalise_cfd_round
    frame["cfd_round"] = [normalise_cfd_round(r, c) for r, c in zip(frame["cfd_round"], frame["carrier"])]
    frame["unit_id"] = frame["CFD_ID"].astype(str)
    frame["unit_name"] = frame["Name_of_CFD_Unit"].astype(str)
    frame["unit_capacity_mw"] = pd.to_numeric(frame["Maximum_Contract_Capacity_MW"], errors="coerce")
    frame["postcode"] = ""
    frame["award_year"] = frame["cfd_round"].map(LCCC_ROUND_AWARD_YEAR)
    return frame.reset_index(drop=True)


def load_repd_location(repd_path) -> pd.DataFrame:
    """REPD postcode and country by Ref ID (the site files carry neither)."""
    repd = pd.read_csv(repd_path, encoding="latin1", low_memory=False,
                       usecols=["Ref ID", "Post Code", "Country"])
    repd["repd_ref_id"] = repd["Ref ID"].map(clean_reference)
    repd["site_postcode"] = repd["Post Code"].map(postcode)
    repd["site_country"] = repd["Country"].fillna("").astype(str).str.strip()
    return repd.drop_duplicates("repd_ref_id").set_index("repd_ref_id")[["site_postcode", "site_country"]]


OVERRIDE_COLUMNS = ["register", "unit_id", "repd_ref_id", "action", "unit_name", "site_name", "evidence"]


def load_crosswalk_overrides(path) -> pd.DataFrame:
    """Sourced explicit pairs: ``unit_id`` (Ofgem reference or LCCC CFD_ID) to REPD Ref ID.

    ``register`` is ``ofgem_ro`` or ``lccc_cfd``. ``action = match`` pairs the
    unit with the listed REPD site (one row per site for a unit spanning
    several); ``action = no_match`` records a reviewed unit that has no named
    site, so it cannot be auto-matched to a namesake.
    """
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    frame = pd.read_csv(path, dtype=str, comment="#").fillna("")
    required = {"register", "unit_id", "repd_ref_id", "action", "evidence"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Support crosswalk overrides are missing columns: {sorted(missing)}")
    unknown = sorted(set(frame["action"]) - {"match", "no_match"})
    if unknown:
        raise ValueError(f"Support crosswalk override actions must be match/no_match, not {unknown}")
    registers = sorted(set(frame["register"]) - {"ofgem_ro", "lccc_cfd"})
    if registers:
        raise ValueError(f"Support crosswalk override registers must be ofgem_ro/lccc_cfd, not {registers}")
    if frame["evidence"].str.strip().eq("").any():
        raise ValueError("Every support crosswalk override needs an evidence entry")
    frame["unit_id"] = frame["unit_id"].map(clean_reference)
    frame["repd_ref_id"] = frame["repd_ref_id"].map(clean_reference)
    return frame


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

MAPPING_COLUMNS = ["unit_id", "site", "repd_ref_id", "method", "name_score", "postcode_agrees",
                   "unit_capacity_mw", "site_capacity_mw", "allocated_mw"]


def match_units_to_sites(units: pd.DataFrame, sites: pd.DataFrame,
                         overrides: Optional[pd.DataFrame] = None,
                         register: str = "ofgem_ro") -> pd.DataFrame:
    """Match register units to named sites; one row per (unit, site) pair.

    ``units`` needs unit_id, unit_name, carrier, unit_capacity_mw and optionally
    postcode, unit_year (Ofgem accreditation year) and award_year (LCCC round
    award year). ``sites`` is indexed by site row label with name, carrier,
    capacity_mw, repd_ref_id and optionally site_postcode and operating_year.
    ``allocated_mw`` splits a unit spanning several sites in proportion to site
    capacity.

    Timing gates every automatic stage: an Ofgem station matches no site that
    began operating more than three years after its accreditation, and one
    accredited from 2005 (when accreditation followed commissioning) none that
    began more than three years before it; an LCCC contract matches only sites
    operating no earlier than its round was awarded. Stages, each
    only on what earlier stages left:

    1. sourced overrides;
    2. equal phase-free name keys: one unit to several phase sites (``Race
       Bank`` to its two REPD phases) or several units to one site where the
       totals agree, a single pair also where the unit is a part of its site or
       at most twice it, when the name is distinctive; a name shared by several
       units and sites ("Manor Farm") only one to one on postcode or on
       capacity within 5 per cent;
    3. fuzzy, strict: a distinctive shared name token (or a matching postcode)
       with compatible phase markers and agreeing capacity;
    4. fuzzy, groups: one unit to several sites sharing its distinctive name or
       its postcode (``Mid Hill`` to ``Mid Hill I`` and ``Mid Hill Phase II``),
       where the totals agree;
    5. fuzzy, loose: a distinctive shared name with the unit a part of its site
       or at most twice it.
    """
    if units.empty or sites.empty:
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    units = units.copy()
    sites = sites.copy()
    units["postcode"] = units["postcode"].fillna("").astype(str) if "postcode" in units else ""
    for column in ("award_year", "unit_year"):
        if column not in units:
            units[column] = np.nan
    sites["site_postcode"] = sites["site_postcode"].fillna("").astype(str) if "site_postcode" in sites else ""
    if "operating_year" not in sites:
        sites["operating_year"] = np.nan
    sites["operating_year"] = pd.to_numeric(sites["operating_year"], errors="coerce")
    units["_key"] = units["unit_name"].map(phase_free_key)
    units["_name"] = units["unit_name"].map(normalise_station_name)
    units["_marker"] = units["unit_name"].map(phase_marker)
    units["_distinct"] = units["_key"].map(lambda k: bool(distinctive_tokens(k)))
    sites["_key"] = sites["name"].map(phase_free_key)
    sites["_name"] = sites["name"].map(normalise_station_name)
    sites["_marker"] = sites["name"].map(phase_marker)
    sites["repd_ref_id"] = sites["repd_ref_id"].map(clean_reference)
    unit_by_id = units.drop_duplicates("unit_id").set_index("unit_id")

    rows: list = []
    assigned_units: set = set()
    site_load = pd.Series(0.0, index=sites.index)

    def add_pairs(unit_ids, site_labels, method, name_score=None, postcode_agrees=False):
        site_caps = sites.loc[site_labels, "capacity_mw"].astype(float)
        total_site = float(site_caps.sum())
        for unit_id in unit_ids:
            unit_cap = float(unit_by_id.at[unit_id, "unit_capacity_mw"])
            for label, site_cap in site_caps.items():
                share = site_cap / total_site if total_site > 0 else 1.0 / len(site_caps)
                allocated = unit_cap * share
                site_load.at[label] += allocated
                rows.append({
                    "unit_id": unit_id, "site": label,
                    "repd_ref_id": sites.at[label, "repd_ref_id"], "method": method,
                    "name_score": name_score, "postcode_agrees": bool(postcode_agrees),
                    "unit_capacity_mw": unit_cap, "site_capacity_mw": float(site_cap),
                    "allocated_mw": allocated,
                })
            assigned_units.add(unit_id)

    def timing_ok(unit, candidates: pd.DataFrame) -> pd.Series:
        year = candidates["operating_year"]
        ok = pd.Series(True, index=candidates.index)
        award, accredited = unit.get("award_year"), unit.get("unit_year")
        if pd.notna(award):
            ok &= year.isna() | year.ge(float(award))
        if pd.notna(accredited):
            # A station is accredited when it operates, never years before.
            ok &= year.isna() | year.le(float(accredited) + 3)
            if float(accredited) >= 2005:
                ok &= year.isna() | year.ge(float(accredited) - 3)
        return ok

    def free_sites(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[site_load.reindex(frame.index).eq(0.0)]

    # 1. Sourced overrides.
    if overrides is not None and not overrides.empty:
        chosen = overrides[overrides["register"].eq(register)]
        for unit_id, group in chosen.groupby("unit_id", sort=False):
            if unit_id not in unit_by_id.index:
                continue
            if group["action"].eq("no_match").any():
                assigned_units.add(unit_id)
                continue
            labels = sites.index[sites["repd_ref_id"].isin(set(group["repd_ref_id"]))].tolist()
            if labels:
                add_pairs([unit_id], labels, "explicit_override", 100.0)

    searchable = units[~units["unit_id"].isin(assigned_units)
                       & units["unit_capacity_mw"].ge(MIN_MATCH_CAPACITY_MW)
                       & units["_key"].ne("")]

    # 2. Equal phase-free keys.
    for (carrier, key), group in searchable.groupby(["carrier", "_key"], sort=False):
        group = group[~group["unit_id"].isin(assigned_units)]
        if group.empty:
            continue
        candidates = free_sites(sites[sites["carrier"].eq(carrier) & sites["_key"].eq(key)])
        if len(group) == 1:
            candidates = candidates[timing_ok(group.iloc[0], candidates)]
        if candidates.empty:
            continue
        distinct = bool(group["_distinct"].iloc[0])
        unit_total = float(group["unit_capacity_mw"].sum())
        site_total = float(candidates["capacity_mw"].sum())
        if distinct and (len(group) == 1 or len(candidates) == 1):
            part_of_site = (len(candidates) == 1 and unit_total
                            <= site_total * SITE_CAPACITY_HEADROOM + CAPACITY_ABSOLUTE_TOLERANCE_MW)
            larger = (len(candidates) == 1 and len(group) == 1
                      and unit_total <= MAX_UNIT_TO_SITE_RATIO * site_total)
            if capacity_agrees(unit_total, site_total) or part_of_site or larger:
                agrees = any(p and p in set(candidates["site_postcode"]) for p in group["postcode"])
                add_pairs(group["unit_id"].tolist(), candidates.index.tolist(),
                          "exact_name_group", 100.0, agrees)
            continue
        # A generic or repeated name: one to one on postcode, then on capacity
        # within 5 per cent.
        for _, unit in group.sort_values("unit_capacity_mw", ascending=False).iterrows():
            free = free_sites(candidates)
            free = free[timing_ok(unit, free)]
            if free.empty:
                break
            pick, agrees = None, False
            by_postcode = free[free["site_postcode"].ne("") & free["site_postcode"].eq(unit["postcode"])]
            if len(by_postcode) == 1 and capacity_agrees(unit["unit_capacity_mw"], by_postcode["capacity_mw"].iloc[0]):
                pick, agrees = by_postcode.index[0], True
            elif len(by_postcode) == 0:
                close = free[[capacity_agrees(unit["unit_capacity_mw"], c, relative=0.05, absolute=0.05)
                              for c in free["capacity_mw"]]]
                if len(close) == 1:
                    pick = close.index[0]
            if pick is not None:
                add_pairs([unit["unit_id"]], [pick], "exact_name_one_to_one", 100.0, agrees)

    sites_by_carrier = {carrier: group for carrier, group in sites.groupby("carrier")}
    # Candidate index: sites sharing a four-letter token prefix (which keeps
    # near-spellings such as Brockloch/Brochloch) or the postcode.
    prefix_index: dict = {}
    for label, key, code in zip(sites.index, sites["_key"], sites["site_postcode"]):
        for token in set(key.split()):
            if len(token) >= 3 and not token.isdigit():
                prefix_index.setdefault(token[:4], set()).add(label)
        if code:
            prefix_index.setdefault(("postcode", code), set()).add(label)

    def evidence(unit, pool: pd.DataFrame) -> dict:
        overlap = np.array([distinctive_overlap(unit["_key"], key) for key in pool["_key"]], dtype=bool)
        contained = np.array([distinctive_overlap(unit["_key"], key, directional=True)
                              for key in pool["_key"]], dtype=bool)
        postcode_hit = (pool["site_postcode"].ne("") & pool["site_postcode"].eq(unit["postcode"])).to_numpy()
        sort_scores = np.array([fuzz.token_sort_ratio(unit["_name"], name) for name in pool["_name"]])
        set_scores = np.array([fuzz.token_set_ratio(unit["_key"], key) for key in pool["_key"]])
        markers = np.array([markers_compatible(unit["_marker"], m) for m in pool["_marker"]], dtype=bool)
        return {"overlap": overlap, "contained": contained, "postcode": postcode_hit, "markers": markers,
                "score": np.maximum(sort_scores, set_scores)}

    def candidates_for(unit) -> pd.DataFrame:
        pool = sites_by_carrier.get(unit["carrier"])
        if pool is None or pool.empty:
            return pd.DataFrame()
        labels = set()
        for token in set(unit["_key"].split()):
            if len(token) >= 3 and not token.isdigit():
                labels |= prefix_index.get(token[:4], set())
        if unit["postcode"]:
            labels |= prefix_index.get(("postcode", unit["postcode"]), set())
        pool = pool[pool.index.isin(labels)]
        return pool[timing_ok(unit, pool)]

    def pick_single(unit, strict: bool):
        pool = candidates_for(unit)
        if pool.empty:
            return None
        cap = float(unit["unit_capacity_mw"])
        load = site_load.reindex(pool.index)
        room = pool["capacity_mw"] * SITE_CAPACITY_HEADROOM + CAPACITY_ABSOLUTE_TOLERANCE_MW - load
        fresh = load.eq(0.0).to_numpy()
        if strict:
            keep = fresh & np.array([capacity_agrees(cap, c) for c in pool["capacity_mw"]], dtype=bool)
        else:
            keep = room.ge(cap).to_numpy() | (
                fresh & (pool["capacity_mw"] * MAX_UNIT_TO_SITE_RATIO).ge(cap).to_numpy())
        pool = pool[keep]
        if pool.empty:
            return None
        ev = evidence(unit, pool)
        blank_marker = np.array([unit["_marker"] == "" or m == "" for m in pool["_marker"]], dtype=bool)
        if strict:
            # Agreeing capacity and timing: a shared distinctive name, or a
            # full postcode even where the company and site names differ
            # (Lochlebank / Lochelbank Wind Farm II). An unnumbered name may
            # stand for a numbered record ("Spurness Wind Farm II").
            accept = (ev["overlap"] & (ev["markers"] | blank_marker)) | ev["postcode"]
        else:
            # A part-capacity match needs the whole unit name in the site name
            # ("Clyde Windfarm (North)" in "Clyde Wind Farm") or, for a site
            # not yet matched, the site name in the unit name ("Toggam Farm"
            # in "Greenheath Toggam Solar PV"); a site already matched takes
            # a further unit only on the first.
            accept = (ev["contained"] | (ev["overlap"] & fresh[keep])) & ev["markers"]
        if not accept.any():
            return None
        closeness = np.array([1.0 - min(abs(cap - c) / max(cap, c), 1.0) for c in pool["capacity_mw"]])
        joint = (0.75 * ev["score"] + 25.0 * ev["postcode"] + 15.0 * closeness + 5.0 * ev["overlap"]
                 + 5.0 * ev["markers"])
        joint = np.where(accept, joint, -1.0)
        best = int(np.argmax(joint))
        return pool.index[best], float(ev["score"][best]), bool(ev["postcode"][best])

    def pick_group(unit):
        pool = free_sites(candidates_for(unit))
        if len(pool) < 2:
            return None
        ev = evidence(unit, pool)
        for basis, mask in (("name", ev["overlap"]), ("postcode", ev["postcode"])):
            members = pool[mask]
            if len(members) >= 2 and capacity_agrees(float(unit["unit_capacity_mw"]),
                                                     float(members["capacity_mw"].sum())):
                return members.index.tolist(), basis
        return None

    # 3 to 5. Fuzzy stages, largest units first.
    for stage in ("strict", "group", "loose"):
        remaining = searchable[~searchable["unit_id"].isin(assigned_units)].sort_values(
            "unit_capacity_mw", ascending=False)
        for _, unit in remaining.iterrows():
            if stage == "group":
                found = pick_group(unit)
                if found is not None:
                    add_pairs([unit["unit_id"]], found[0], f"fuzzy_group_{found[1]}", None,
                              found[1] == "postcode")
                continue
            found = pick_single(unit, strict=stage == "strict")
            if found is not None:
                add_pairs([unit["unit_id"]], [found[0]], f"fuzzy_{stage}", found[1], found[2])
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def _operating_year(sites_df: pd.DataFrame) -> pd.Series:
    if "operational_date" not in sites_df.columns:
        return pd.Series(np.nan, index=sites_df.index)
    dates = pd.to_datetime(sites_df["operational_date"], dayfirst=True, errors="coerce", format="mixed")
    return dates.dt.year


def _site_table(sites_df: pd.DataFrame, mask: pd.Series, repd_location: Optional[pd.DataFrame]) -> pd.DataFrame:
    part = sites_df.loc[mask]
    if "source_site_name" in part.columns:
        name = part["source_site_name"]
        if "site_name" in part.columns:
            name = name.fillna(part["site_name"])
    else:
        name = part["site_name"]
    table = pd.DataFrame({
        "name": name.fillna("").astype(str),
        "carrier": part["technology"],
        "capacity_mw": pd.to_numeric(part["capacity_mw"], errors="coerce").fillna(0.0),
        "repd_ref_id": part["repd_ref_id"].map(clean_reference) if "repd_ref_id" in part else "",
        "operating_year": _operating_year(part),
    }, index=part.index)
    if repd_location is not None and not repd_location.empty:
        table = table.join(repd_location, on="repd_ref_id")
    for column in ("site_postcode", "site_country"):
        if column not in table:
            table[column] = ""
        table[column] = table[column].fillna("").astype(str)
    if "country" in part.columns:
        own = part["country"].fillna("").astype(str)
        table["site_country"] = table["site_country"].where(table["site_country"].ne(""), own)
    return table


def _named_mask(sites_df: pd.DataFrame) -> pd.Series:
    origin = sites_df["fleet_origin"] if "fleet_origin" in sites_df else pd.Series("named", index=sites_df.index)
    return origin.eq("named") & sites_df["technology"].isin(SITE_CARRIERS)


def _named_operational_mask(sites_df: pd.DataFrame) -> pd.Series:
    """Named REPD sites that were operating, the only ones that can hold ROCs."""
    status = (sites_df["development_status"] if "development_status" in sites_df
              else pd.Series("Operational", index=sites_df.index))
    return _named_mask(sites_df) & status.fillna("Operational").eq("Operational")


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def _joined(values) -> str:
    return ";".join(sorted({str(v) for v in values if pd.notna(v) and str(v)}))


def apply_lccc_cfd_evidence(sites_df: pd.DataFrame, portfolio: pd.DataFrame,
                            repd_location: Optional[pd.DataFrame] = None,
                            overrides: Optional[pd.DataFrame] = None,
                            logger: logging.Logger = LOGGER):
    """Fill LCCC contract evidence into named sites before the subsidy assignment.

    For each named site matched to a live or pre-start LCCC contract:

    * REPD gives a round but no capacity (Triton Knoll, Eastfields): the LCCC
      contract capacity is filled in, so the REPD branch labels the site CfD;
    * REPD gives no round: the LCCC round and capacity are filled in;
    * REPD gives a round and a capacity: nothing changes; a different LCCC
      round is reported, not applied.

    A REPD round with no capacity that LCCC does not confirm (a round entered
    but not awarded, as REPD also records: Docking Farm Solar) is withdrawn and
    kept in ``repd_cfd_round_unconfirmed``, so the site is routed like any
    project without recorded support. Returns (sites, pair mapping, audit).
    """
    result = sites_df.copy()
    for column in ("cfd_round", "cfd_id", "cfd_support_source", "cfd_technology"):
        if column not in result.columns:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="object")
        else:
            result[column] = result[column].astype("object")
    if "cfd_capacity_mw" not in result.columns:
        result["cfd_capacity_mw"] = np.nan
    mask = _named_mask(result)
    table = _site_table(result, mask, repd_location)
    mapping = match_units_to_sites(portfolio, table, overrides, register="lccc_cfd")
    audit_rows = []
    matched_sites = set()
    if not mapping.empty:
        columns = ["unit_id", "unit_name", "cfd_round", "Status"]
        if "Technology_Type" in portfolio.columns:
            columns.append("Technology_Type")
        mapping = mapping.merge(portfolio[columns], on="unit_id", how="left")
        for label, pairs in mapping.groupby("site"):
            matched_sites.add(label)
            rounds = sorted(set(pairs["cfd_round"]))
            # A site matched to contracts of several rounds settles at the
            # round holding most of its matched MW: a joined label such as
            # "AR6/AR7" is not a strike-table key and fell back to the future
            # strike (West Burton Solar, audit 27 Sep 2026). The audit keeps
            # every round in ``round_lccc``.
            by_round = pairs.groupby("cfd_round")["allocated_mw"].sum()
            lccc_round = rounds[0] if len(rounds) == 1 else str(
                by_round.sort_values(ascending=False, kind="stable").index[0])
            lccc_rounds_all = "/".join(rounds)
            repd_round = result.at[label, "cfd_round"]
            repd_round = str(repd_round).strip() if pd.notna(repd_round) else ""
            repd_capacity = pd.to_numeric(pd.Series([result.at[label, "cfd_capacity_mw"]]), errors="coerce").iloc[0]
            contracted = float(min(pairs["allocated_mw"].sum(), table.at[label, "capacity_mw"]))
            reviewed = pairs["method"].eq("explicit_override").all()
            if repd_round and pd.notna(repd_capacity) and repd_capacity > 0 and reviewed \
                    and repd_round != lccc_round:
                # A sourced crosswalk pair is reviewed evidence: its contracts'
                # round replaces REPD's (Hornsea 3: REPD AR6, LCCC AR4 2,146 MW
                # + AR6 1,080 MW; audit 27 Sep 2026).
                result.at[label, "cfd_round"] = lccc_round
                result.at[label, "cfd_capacity_mw"] = contracted
                result.at[label, "cfd_support_source"] = "lccc_cfd_register"
                note = "reviewed override: LCCC round replaces REPD round"
            elif repd_round and pd.notna(repd_capacity) and repd_capacity > 0:
                note = "round confirmed" if repd_round == lccc_round else "REPD round kept; LCCC round differs"
            else:
                if not repd_round:
                    result.at[label, "cfd_round"] = lccc_round
                    note = "round and capacity from LCCC"
                else:
                    note = ("capacity from LCCC (REPD round confirmed)" if repd_round == lccc_round
                            else "capacity from LCCC; REPD round kept, LCCC round differs")
                result.at[label, "cfd_capacity_mw"] = contracted
                result.at[label, "cfd_support_source"] = "lccc_cfd_register"
            result.at[label, "cfd_id"] = _joined(pairs["unit_id"])
            # The contract's strike technology where its units are floating or remote
            # island wind (all the units of the round the site settles at).
            if "Technology_Type" in pairs.columns:
                own = pairs[pairs["cfd_round"].eq(result.at[label, "cfd_round"])]
                kinds = set(own["Technology_Type"].map(LCCC_STRIKE_TECHNOLOGY).fillna(""))
                if len(kinds) == 1 and next(iter(kinds)):
                    result.at[label, "cfd_technology"] = next(iter(kinds))
            audit_rows.append({
                "site": label, "site_name": table.at[label, "name"], "carrier": table.at[label, "carrier"],
                "capacity_mw": table.at[label, "capacity_mw"], "register": "lccc_cfd",
                "units": _joined(pairs["unit_id"]), "unit_names": _joined(pairs["unit_name"]),
                "method": _joined(pairs["method"]), "matched_mw": float(pairs["allocated_mw"].sum()),
                "round_repd": repd_round, "round_lccc": lccc_rounds_all,
                "round_after": result.at[label, "cfd_round"],
                "cfd_capacity_after": result.at[label, "cfd_capacity_mw"], "note": note,
            })
    round_only = mask & result["cfd_round"].fillna("").astype(str).str.strip().ne("") \
        & pd.to_numeric(result["cfd_capacity_mw"], errors="coerce").isna()
    if "repd_cfd_round_unconfirmed" not in result.columns:
        result["repd_cfd_round_unconfirmed"] = pd.Series(pd.NA, index=result.index, dtype="object")
    for label in result.index[round_only]:
        if label in matched_sites:
            continue
        # No contract: the round is withdrawn (kept for audit) so the site is
        # routed like any project without recorded support.
        repd_round = result.at[label, "cfd_round"]
        result.at[label, "repd_cfd_round_unconfirmed"] = repd_round
        result.at[label, "cfd_round"] = pd.NA
        audit_rows.append({
            "site": label, "site_name": table.at[label, "name"], "carrier": table.at[label, "carrier"],
            "capacity_mw": table.at[label, "capacity_mw"], "register": "lccc_cfd",
            "units": "", "unit_names": "", "method": "none", "matched_mw": 0.0,
            "round_repd": repd_round, "round_lccc": "", "round_after": "",
            "cfd_capacity_after": np.nan,
            "note": "REPD round without capacity and no live LCCC contract: round withdrawn",
        })
    audit = pd.DataFrame(audit_rows)
    filled = audit[audit["note"].str.contains("from LCCC")] if len(audit) else audit
    logger.info(
        "LCCC CfD register: %d contract units matched to %d named sites; %d sites (%.0f MW) "
        "take LCCC round/capacity evidence",
        mapping["unit_id"].nunique() if not mapping.empty else 0, len(matched_sites),
        len(filled), float(filled["capacity_mw"].sum()) if len(filled) else 0.0,
    )
    return result, mapping, audit


def apply_ro_register_labels(sites_df: pd.DataFrame, register: pd.DataFrame,
                             repd_location: Optional[pd.DataFrame] = None,
                             overrides: Optional[pd.DataFrame] = None,
                             logger: logging.Logger = LOGGER):
    """Relabel named operational sites from the Ofgem RO register.

    Run after the subsidy assignment. For each named operational wind or solar
    site:

    * matched to accreditations still running on the reference date: ROC, with
      the capacity-weighted banding of those stations and their references; a
      site whose unaccredited remainder exceeds 20 per cent and 1 MW is split
      into a ROC row and a merchant row (same ``site_name``, so both keep the
      site's profile);
    * a CfD site stays CfD whatever its accreditation (codebase precedence);
    * matched only to expired accreditations, or not matched at all: not ROC.
      The date-based inference is withdrawn for such sites. The register lists
      every RO station, so a site absent from it holds no ROCs (a Feed-in
      Tariff or unsupported project).

    Returns (sites, pair mapping, per-site audit).
    """
    result = sites_df.copy()
    if "ro_banding" not in result.columns:
        result["ro_banding"] = np.nan
    result["ro_banding"] = pd.to_numeric(result["ro_banding"], errors="coerce")
    for column in ("ro_accreditation_reference", "ro_support_source"):
        if column not in result.columns:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="object")
        else:
            result[column] = result[column].astype("object")
    mask = _named_operational_mask(result)
    table = _site_table(result, mask, repd_location)
    units = register[register["gb"] & register["carrier"].isin(SITE_CARRIERS)]
    mapping = match_units_to_sites(units, table, overrides, register="ofgem_ro")
    if not mapping.empty:
        mapping = mapping.merge(
            units[["unit_id", "unit_name", "ro_eligible", "banding", "banding_source",
                   "effective_date", "roc_expiry_capped"]],
            on="unit_id", how="left")
    by_site = {label: pairs for label, pairs in mapping.groupby("site")} if not mapping.empty else {}

    audit_rows, extra_rows = [], []
    for label in table.index:
        before_support = result.at[label, "support_type"]
        before_banding = result.at[label, "ro_banding"]
        capacity = float(table.at[label, "capacity_mw"])
        pairs = by_site.get(label)
        eligible = pairs[pairs["ro_eligible"].astype(bool)] if pairs is not None else None
        eligible_mw = float(eligible["allocated_mw"].sum()) if eligible is not None else 0.0
        expired_mw = float(pairs["allocated_mw"].sum()) - eligible_mw if pairs is not None else 0.0
        after_support, after_banding, roc_mw = before_support, before_banding, 0.0
        if before_support == "CfD":
            note = "CfD precedence over an RO accreditation" if eligible_mw > 0 else "CfD"
        elif eligible_mw > 0:
            roc_mw = min(eligible_mw, capacity)
            banding = float((eligible["allocated_mw"] * eligible["banding"]).sum() / eligible_mw)
            after_support, after_banding = "ROC", round(banding, 4)
            result.at[label, "support_type"] = "ROC"
            result.at[label, "ro_banding"] = after_banding
            result.at[label, "ro_accreditation_reference"] = _joined(eligible["unit_id"])
            result.at[label, "ro_support_source"] = "ofgem_ro_register"
            remainder = capacity - roc_mw
            share = SPLIT_MIN_REMAINDER_SHARE_BY_CARRIER.get(table.at[label, "carrier"], SPLIT_MIN_REMAINDER_SHARE)
            if remainder > SPLIT_MIN_REMAINDER_MW and remainder > share * capacity:
                result.at[label, "capacity_mw"] = roc_mw
                extra = result.loc[label].copy()
                extra["capacity_mw"] = remainder
                extra["support_type"] = "merchant"
                extra["ro_banding"] = np.nan
                extra["ro_accreditation_reference"] = pd.NA
                extra["ro_support_source"] = "ofgem_ro_register_unaccredited_remainder"
                extra_rows.append(extra)
                note = f"split: {roc_mw:.1f} MW ROC + {remainder:.1f} MW merchant"
            else:
                note = "accredited"
        else:
            if before_support == "ROC":
                result.at[label, "support_type"] = "merchant"
                result.at[label, "ro_banding"] = np.nan
                after_support, after_banding = "merchant", np.nan
            result.at[label, "ro_support_source"] = (
                "ofgem_ro_register_expired" if pairs is not None else "ofgem_ro_register_no_accreditation")
            note = "accreditation expired" if pairs is not None else "no accreditation found"
        audit_rows.append({
            "site": label, "site_name": table.at[label, "name"], "carrier": table.at[label, "carrier"],
            "capacity_mw": capacity, "register": "ofgem_ro",
            "units": _joined(pairs["unit_id"]) if pairs is not None else "",
            "unit_names": _joined(pairs["unit_name"]) if pairs is not None else "",
            "method": _joined(pairs["method"]) if pairs is not None else "none",
            "matched_mw": float(pairs["allocated_mw"].sum()) if pairs is not None else 0.0,
            "eligible_mw": eligible_mw, "expired_mw": expired_mw, "roc_mw": roc_mw,
            "support_before": before_support, "support_after": after_support,
            "banding_before": before_banding, "banding_after": after_banding,
            "banding_source": _joined(eligible["banding_source"]) if eligible is not None else "",
            "note": note,
        })
    if extra_rows:
        result = pd.concat([result, pd.DataFrame(extra_rows)], ignore_index=True, sort=False)
    audit = pd.DataFrame(audit_rows)
    for carrier, part in audit.groupby("carrier") if len(audit) else []:
        before = float(part.loc[part["support_before"].eq("ROC"), "capacity_mw"].sum())
        logger.info("Ofgem RO register, %s: ROC %.0f MW before, %.0f MW after (%d named operational sites)",
                    carrier, before, float(part["roc_mw"].sum()), len(part))
    return result, mapping, audit


# ---------------------------------------------------------------------------
# Checked CfD contracts (F19 and its follow-up, rebuild batch of 13 Oct 2026)
# ---------------------------------------------------------------------------

# REPD's "CfD Capacity" often holds one round of several (Sofia 14 of 1,400 MW,
# Inch Cape 266 of 1,076, Viking 223.6 of 443.6), REPD keeps rounds whose
# contracts LCCC has terminated (Hornsea 4, Norfolk Boreas), and the register
# matcher's phase matches are partial or wrong (Dudgeon Extension took two of
# the existing Dudgeon farm's phases; EA3 took East Anglia Two Phase 3). So a
# site's contracts come from a declared table checked unit by unit against the
# LCCC portfolio, one row per (site, round), and the build applies it:
#
# * a part-contracted site (Seagreen 454 of 1,075 MW) splits into its CfD
#   rounds and a merchant remainder;
# * a site whose contracts span rounds (Moray West AR4 220.5 + AR6 73.5) splits
#   so each part settles at its own round's strike;
# * a site whose every contract is Terminated, or which holds none, carries
#   ``action: not_built_refill`` (user decision, 14 Oct 2026): it is treated as
#   not built by 2030, the CP30 brownfield fleet drops it through the named-fleet
#   exclusions (brownfield_fleet.apply_named_fleet_exclusions), and the ES5 proxy
#   residual refills its MW in the same region and technology as new 2030
#   capacity (FES future-CfD fractions, the unassigned future cohort). Where it
#   is still present (a later model year, or a build without the brownfield
#   fleet) it has no CfD.
CHECKED_CFD_COLUMNS = ("repd_ref_id", "site_name", "cfd_round", "contracted_mw", "cfd_ids", "lccc_status", "source")
#: ``action`` values: apply the contracts; ``terminated_operating`` (the plant
#: operates but its contract is terminated: it stays in the fleet without a CfD,
#: merchant or ROC by the RO register, e.g. Netley Landfill Solar since 2017); or
#: treat the site as not built up to the year (excluded from the named fleet and
#: refilled by the proxy).
APPLY_CONTRACT_ACTION = "apply_contract"
TERMINATED_OPERATING_ACTION = "terminated_operating"
NOT_BUILT_ACTIONS = {"not_built_refill": 2030}
#: Convention for contracts LCCC expects to start after the model year (user
#: decision, 14 Oct 2026): where CP30 counts the capacity as built (Inch Cape's AR4
#: phases 2-3 from March 2031, Pentland, AR7 phases of 2031-32), it settles on its
#: OWN contract round and strike, as a CfD may start early within its window. The
#: table therefore records a contract whatever its expected start date, and the
#: settlement applies no start-date gate to it.
#: ``lccc_status`` of a site with no contract at all (REPD or the matcher gave it one).
NO_CONTRACT_STATUS = "none"
TERMINATED_STATUS = "Terminated"
# Generator name suffixes. The CfD round holding most MW keeps the site's own
# name (so name-keyed lookups such as the LCCC contract starts still find it);
# another round's part is "<site>_CfD_part_<round>" and the uncontracted
# remainder "<site>_non_CfD". Every part keeps the site's source_site_name, so
# it takes the site's profile.
PARTIAL_CFD_REMAINDER_SUFFIX = "_non_CfD"
CFD_ROUND_PART_SUFFIX = "_CfD_part_"
CFD_CONTRACT_PART_COLUMN = "cfd_contract_part"
_PART_SUFFIX_RE = re.compile(rf"({re.escape(PARTIAL_CFD_REMAINDER_SUFFIX)}|{re.escape(CFD_ROUND_PART_SUFFIX)}.+)$")


def partial_cfd_base_name(name: object) -> str:
    """A generator or site name without a checked-contract part suffix.

    Name-keyed lookups (a farm's offshore local TNUoS tariff) treat every part
    of a split site as the site.
    """
    return _PART_SUFFIX_RE.sub("", str(name))


def load_checked_cfd_contracts(path) -> pd.DataFrame:
    """The checked CfD contract table: one row per (REPD Ref ID, round)."""
    table = pd.read_csv(path, dtype={"repd_ref_id": str, "cfd_round": str, "cfd_ids": str})
    if missing := set(CHECKED_CFD_COLUMNS).difference(table.columns):
        raise ValueError(f"Checked CfD contract file {path} is missing: {sorted(missing)}")
    table["repd_ref_id"] = table["repd_ref_id"].map(clean_reference)
    table["cfd_round"] = table["cfd_round"].fillna("").astype(str).str.strip()
    table["cfd_ids"] = table["cfd_ids"].fillna("").astype(str)
    table["lccc_status"] = table["lccc_status"].fillna("").astype(str).str.strip()
    table["contracted_mw"] = pd.to_numeric(table["contracted_mw"], errors="raise")
    if "action" not in table.columns:
        table["action"] = APPLY_CONTRACT_ACTION
    table["action"] = table["action"].fillna("").astype(str).str.strip().replace("", APPLY_CONTRACT_ACTION)
    if unknown := sorted(set(table["action"]) - {APPLY_CONTRACT_ACTION, TERMINATED_OPERATING_ACTION,
                                                 *NOT_BUILT_ACTIONS}):
        raise ValueError(f"Checked CfD contract file {path} has unknown actions {unknown}")
    if "reason" not in table.columns:
        table["reason"] = ""
    table["reason"] = table["reason"].fillna("").astype(str)
    if "cfd_technology" not in table.columns:
        table["cfd_technology"] = ""
    table["cfd_technology"] = table["cfd_technology"].fillna("").astype(str).str.strip()
    not_live = table["action"].isin([TERMINATED_OPERATING_ACTION, *NOT_BUILT_ACTIONS])
    if table.loc[not_live, "reason"].str.strip().eq("").any():
        raise ValueError(f"Checked CfD contract file {path}: every not-built or terminated site needs a reason")
    live = ~table["lccc_status"].isin([TERMINATED_STATUS, NO_CONTRACT_STATUS])
    site_live = live.groupby(table["repd_ref_id"]).transform("any")
    if (not_live & site_live).any():
        raise ValueError(f"Checked CfD contract file {path}: a site with a live contract cannot be "
                         "not built or terminated")
    if table.groupby("repd_ref_id")["action"].nunique().gt(1).any():
        raise ValueError(f"Checked CfD contract file {path}: one action per site")
    if table["repd_ref_id"].eq("").any() or table.duplicated(["repd_ref_id", "cfd_round"]).any():
        raise ValueError(f"Checked CfD contract file {path} needs one row per REPD Ref ID and round")
    if table["contracted_mw"].lt(0).any():
        raise ValueError(f"Checked CfD contract file {path} has a negative contracted_mw")
    for ref, group in table.groupby("repd_ref_id"):
        status = set(group["lccc_status"])
        if NO_CONTRACT_STATUS in status and (len(group) > 1 or group["contracted_mw"].gt(0).any()):
            raise ValueError(f"Checked CfD contract file {path}: site {ref} is '{NO_CONTRACT_STATUS}' and has contracts")
        # A terminated round beside a live one (High Constellation: AR4 terminated, AR7a
        # live) is kept for the vintage netting only; the site settles on its live rounds.
    return table


def checked_cfd_exclusions(table: pd.DataFrame) -> pd.DataFrame:
    """The table's not-built sites as named-fleet exclusions, one row per REPD Ref ID.

    In the schema of ``cp30_brownfield.named_exclusions_file`` (repd_ref_id,
    site_name, first_modelled_year, reason, evidence), so the brownfield fleet
    excludes them with that file's rule: out of the named list, their MW refilled
    by the ES5 proxy residual. ``first_modelled_year`` is the year after the
    action's (``not_built_refill``: 2031).
    """
    columns = ["repd_ref_id", "site_name", "first_modelled_year", "reason", "evidence"]
    if table is None or table.empty or "action" not in table.columns:
        return pd.DataFrame(columns=columns)
    active = table[table["action"].isin(list(NOT_BUILT_ACTIONS))]
    rows = []
    for ref, group in active.groupby("repd_ref_id", sort=False):
        first = group.iloc[0]
        ids = ";".join(i for i in group["cfd_ids"] if i)
        rows.append({"repd_ref_id": ref, "site_name": first["site_name"],
                     "first_modelled_year": NOT_BUILT_ACTIONS[first["action"]] + 1, "reason": first["reason"],
                     "evidence": f"LCCC {first['lccc_status']}" + (f" ({ids})" if ids else "")})
    return pd.DataFrame(rows, columns=columns)


def terminated_round_capacity(table: pd.DataFrame) -> pd.DataFrame:
    """Awarded MW per (carrier, round) the table records as Terminated, for the vintage netting.

    The CfD vintage schedule is the capacity each round awarded; the proxy
    increment fills what the named fleet does not hold. A named site whose
    contract is terminated no longer holds its round, so without this its
    awarded MW would pass to the increment at the terminated round's strike.
    Rows with ``net_from_vintage_schedule`` false (AR4 Remote Island Wind, not
    in the onshore schedule) are left out.
    """
    columns = ["carrier", "cfd_round", "capacity_mw"]
    if table is None or table.empty or "carrier" not in table.columns:
        return pd.DataFrame(columns=columns)
    terminated = table["lccc_status"].eq(TERMINATED_STATUS)
    if "net_from_vintage_schedule" in table.columns:
        flag = table["net_from_vintage_schedule"].astype(str).str.strip().str.lower()
        terminated &= ~flag.isin(["false", "0", "no"])
    rows = table.loc[terminated, ["carrier", "cfd_round", "contracted_mw"]].rename(
        columns={"contracted_mw": "capacity_mw"})
    return rows.groupby(["carrier", "cfd_round"], as_index=False)["capacity_mw"].sum()


def apply_checked_cfd_contracts(sites_df: pd.DataFrame, table: pd.DataFrame,
                                logger: logging.Logger = LOGGER):
    """Give named CfD sites the contracts the checked table records.

    Run after the subsidy assignment and before the RO register labels. For
    each REPD Ref ID in ``table`` whose named rows are labelled CfD:

    * no live contract (``lccc_status`` Terminated or none): every row becomes
      ``merchant`` with its CfD fields cleared; the RO register then labels it
      as any unsupported named site (ROC where an accreditation covers it).
      This is the ``terminated_operating`` action; a ``not_built_refill`` site
      has already left a 2030 fleet;
    * otherwise each live round takes ``contracted_mw`` (scaled down pro rata
      when the rounds exceed the site), the round holding most MW in place and
      each other round as a ``<site>_CfD_part_<round>`` row, with that round's
      ``cfd_ids`` and ``cfd_technology`` (the contract's strike-table row:
      floating, permitted reduction, Scottish pot, remote island wind; blank for
      the carrier's own); a terminated round beside a live one is ignored here
      and only netted off the vintage schedule. A contract LCCC expects to start
      after the model year still settles on its own round (``NOT_BUILT_ACTIONS``
      note). What is left is a ``merchant`` ``<site>_non_CfD`` row. A
      remainder of 1 MW or less, or under the RO split's share threshold (20
      per cent; 35 for solar, whose REPD capacity may be DC), is not split off:
      the rounds share the whole site in proportion.

    Every part keeps the site's bus and profile, and a site on several rows
    splits each row in proportion, so capacity is conserved exactly. REPD rows
    labelled CfD whose own "CfD Capacity" is below the site's by more than the
    thresholds but which the table does not list are reported, not split.

    Returns (sites, audit).
    """
    result = sites_df.copy()
    if CFD_CONTRACT_PART_COLUMN not in result.columns:
        result[CFD_CONTRACT_PART_COLUMN] = pd.Series(pd.NA, index=result.index, dtype="object")
    if result.empty or "support_type" not in result.columns:
        return result, pd.DataFrame()
    named = (result["fleet_origin"].eq("named") if "fleet_origin" in result.columns
             else pd.Series(True, index=result.index))
    is_cfd = named & result["support_type"].astype(str).str.strip().eq("CfD")
    refs = (result["repd_ref_id"].map(clean_reference) if "repd_ref_id" in result.columns
            else pd.Series("", index=result.index))
    capacity = pd.to_numeric(result["capacity_mw"], errors="coerce").fillna(0.0)
    carrier_col = "technology" if "technology" in result.columns else "carrier"
    cleared = ("cfd_round", "cfd_id", "cfd_support_source", "ro_accreditation_reference", "ro_support_source",
               "cfd_technology")
    for column in cleared + ("source_site_name",):
        if column in result.columns:
            result[column] = result[column].astype("object")
    for column in ("cfd_round", "cfd_id", "cfd_technology"):
        if column not in result.columns:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="object")
    if "cfd_capacity_mw" not in result.columns:
        result["cfd_capacity_mw"] = np.nan

    def merchant(row: pd.Series) -> pd.Series:
        row = row.copy()
        row["support_type"] = "merchant"
        for column in cleared:
            if column in row.index:
                row[column] = pd.NA
        row["cfd_capacity_mw"] = np.nan
        if "ro_banding" in row.index:
            row["ro_banding"] = np.nan
        return row

    def renamed(row: pd.Series, suffix: str) -> pd.Series:
        source_name = row.get("source_site_name")
        if source_name is None or pd.isna(source_name) or not str(source_name).strip():
            row["source_site_name"] = row["site_name"]
        row["site_name"] = f"{row['site_name']}{suffix}"
        return row

    audit_rows, extra_rows = [], []
    for ref, entries in table.groupby("repd_ref_id", sort=False):
        rows = result.index[is_cfd & refs.eq(ref)]
        if not len(rows):
            continue
        site_mw = float(capacity[rows].sum())
        carrier = str(result.at[rows[0], carrier_col])
        record = {"repd_ref_id": ref, "site_name": entries["site_name"].iloc[0], "carrier": carrier,
                  "rows": len(rows), "site_mw": site_mw, "cfd_ids": ";".join(entries["cfd_ids"])}
        live = entries[~entries["lccc_status"].isin([TERMINATED_STATUS, NO_CONTRACT_STATUS])
                       & entries["contracted_mw"].gt(0)]
        if live.empty:
            for label in rows:
                updated = merchant(result.loc[label])
                updated[CFD_CONTRACT_PART_COLUMN] = "no_contract"
                result.loc[label] = updated
            status = ";".join(sorted(set(entries["lccc_status"])))
            audit_rows.append({**record, "rounds": "", "contracted_mw": 0.0, "remainder_mw": site_mw,
                               "note": f"no live CfD (LCCC {status}): {site_mw:.2f} MW merchant"})
            continue
        rounds = live.set_index("cfd_round")["contracted_mw"].astype(float)
        if rounds.sum() > site_mw:
            rounds = rounds * site_mw / rounds.sum()
        remainder = site_mw - float(rounds.sum())
        threshold = SPLIT_MIN_REMAINDER_SHARE_BY_CARRIER.get(carrier, SPLIT_MIN_REMAINDER_SHARE)
        absorbed = not (remainder > SPLIT_MIN_REMAINDER_MW and remainder > threshold * site_mw)
        if absorbed and remainder > 0:
            rounds = rounds * site_mw / rounds.sum()
            remainder = 0.0
        shares = rounds / site_mw
        main = rounds.idxmax()
        ids = live.set_index("cfd_round")["cfd_ids"]
        technologies = live.set_index("cfd_round")["cfd_technology"] if "cfd_technology" in live.columns             else pd.Series("", index=rounds.index)
        for label in rows:
            row_mw = float(capacity[label])
            base = result.loc[label].copy()
            for round_name, share in shares.items():
                part = base.copy() if round_name == main else renamed(base.copy(), f"{CFD_ROUND_PART_SUFFIX}{round_name}")
                part["capacity_mw"] = row_mw * share
                part["cfd_capacity_mw"] = row_mw * share
                part["cfd_round"] = round_name
                part["cfd_id"] = ids[round_name]
                part["cfd_technology"] = technologies[round_name] or pd.NA
                part[CFD_CONTRACT_PART_COLUMN] = "contracted" if round_name == main else "round"
                if round_name == main:
                    result.loc[label] = part
                else:
                    extra_rows.append(part)
            if remainder > 0:
                rest = renamed(merchant(base), PARTIAL_CFD_REMAINDER_SUFFIX)
                rest["capacity_mw"] = row_mw * remainder / site_mw
                rest[CFD_CONTRACT_PART_COLUMN] = "remainder"
                extra_rows.append(rest)
        described = " + ".join(f"{r} {mw:.2f}" for r, mw in rounds.items())
        if len(rounds) == 1 and remainder == 0:
            note = f"wholly CfD: {described} MW"
        else:
            note = f"split: {described} MW CfD" + (f" + {remainder:.2f} MW remainder" if remainder > 0 else "")
        audit_rows.append({**record, "rounds": ";".join(rounds.index), "contracted_mw": float(rounds.sum()),
                           "remainder_mw": remainder, "note": note})

    declared = refs.isin(set(table["repd_ref_id"]))
    repd_cfd = pd.to_numeric(result["cfd_capacity_mw"], errors="coerce")
    gap = capacity - repd_cfd
    threshold = result[carrier_col].astype(str).map(SPLIT_MIN_REMAINDER_SHARE_BY_CARRIER).fillna(SPLIT_MIN_REMAINDER_SHARE)
    unverified = is_cfd & ~declared & gap.gt(SPLIT_MIN_REMAINDER_MW) & gap.gt(threshold * capacity)
    for label in result.index[unverified]:
        audit_rows.append({
            "repd_ref_id": refs[label], "site_name": result.at[label, "site_name"],
            "carrier": result.at[label, carrier_col], "rows": 1, "site_mw": float(capacity[label]),
            "cfd_ids": "", "rounds": "", "contracted_mw": float(repd_cfd[label]), "remainder_mw": 0.0,
            "note": "not split: REPD CfD capacity below the site's, not checked against LCCC",
        })
    if extra_rows:
        result = pd.concat([result, pd.DataFrame(extra_rows)], ignore_index=True, sort=False)
    audit = pd.DataFrame(audit_rows)
    if len(audit):
        note = audit["note"].astype(str)
        split = audit[note.str.startswith("split")]
        withdrawn = audit[note.str.startswith("no live CfD")]
        logger.info(
            "Checked CfD contracts: %d sites split (%.0f MW to merchant remainders), %d sites with no live "
            "CfD (%.0f MW to merchant), %d wholly CfD", len(split), float(split["remainder_mw"].sum()),
            len(withdrawn), float(withdrawn["site_mw"].sum()), int(note.str.startswith("wholly").sum()),
        )
        left = audit[note.str.startswith("not split")]
        if len(left):
            logger.warning(
                "Checked CfD contracts: %d named CfD sites (%.0f MW) have a REPD CfD capacity below the "
                "site's but no checked entry; kept wholly CfD: %s",
                len(left), float(left["site_mw"].sum()), ", ".join(left["site_name"].astype(str).head(12)),
            )
    return result, audit


def register_reconciliation(register: pd.DataFrame, mapping: pd.DataFrame,
                            sites_df: pd.DataFrame) -> pd.DataFrame:
    """ROC-labelled capacity by carrier against the eligible register fleet.

    Columns: register eligible MW (all, GB, Northern Ireland), matched eligible
    MW, labelled ROC MW in the site table, and the unmatched GB remainder split
    at 1 MW (stations below it are mostly not in REPD at all).
    """
    eligible = register[register["ro_eligible"] & register["carrier"].isin(SITE_CARRIERS)].copy()
    matched = set(mapping["unit_id"]) if mapping is not None and not mapping.empty else set()
    eligible["matched"] = eligible["unit_id"].isin(matched)
    labelled = sites_df[sites_df["support_type"].eq("ROC")].groupby("technology")["capacity_mw"].sum()
    rows = []
    for carrier, part in eligible.groupby("carrier"):
        gb = part[part["gb"]]
        unmatched = gb[~gb["matched"]]
        rows.append({
            "carrier": carrier,
            "register_eligible_mw": float(part["capacity_mw"].sum()),
            "register_eligible_gb_mw": float(gb["capacity_mw"].sum()),
            "register_eligible_ni_mw": float(part.loc[~part["gb"], "capacity_mw"].sum()),
            "matched_eligible_mw": float(gb.loc[gb["matched"], "capacity_mw"].sum()),
            "labelled_roc_mw": float(labelled.get(carrier, 0.0)),
            "unmatched_gb_mw": float(unmatched["capacity_mw"].sum()),
            "unmatched_gb_below_1mw_mw": float(unmatched.loc[unmatched["capacity_mw"].lt(1.0), "capacity_mw"].sum()),
            "unmatched_gb_stations": int(len(unmatched)),
        })
    return pd.DataFrame(rows)


def support_register_summary(register: pd.DataFrame, ro_mapping: pd.DataFrame, ro_audit: pd.DataFrame,
                             cfd_audit: pd.DataFrame, sites_df: pd.DataFrame,
                             reference_date, largest: int = 25) -> dict:
    """JSON-ready audit of the register labelling for the build's audit file."""
    reconciliation = register_reconciliation(register, ro_mapping, sites_df)
    changed = ro_audit[ro_audit["support_before"].ne(ro_audit["support_after"])] if len(ro_audit) else ro_audit
    eligible = register[register["ro_eligible"] & register["gb"] & register["carrier"].isin(SITE_CARRIERS)]
    matched = set(ro_mapping["unit_id"]) if ro_mapping is not None and not ro_mapping.empty else set()
    unmatched = eligible[~eligible["unit_id"].isin(matched)].sort_values("capacity_mw", ascending=False)

    def records(frame, columns):
        part = frame.reindex(columns=columns).astype(object)
        return part.where(part.notna(), None).to_dict("records")

    return {
        "method": ("Named operational sites relabelled from the Ofgem RO register (eligibility as the "
                   "bill's RO line) and LCCC contract evidence filled before the subsidy assignment; "
                   "see scripts/renewables/support_register_labels.py"),
        "ro_reference_date": str(pd.Timestamp(reference_date).date()),
        "reconciliation_by_carrier": records(reconciliation, list(reconciliation.columns)),
        "ro_notes": (ro_audit["note"].str.replace(r"split: .*", "split", regex=True).value_counts().to_dict()
                     if len(ro_audit) else {}),
        "ro_changed_sites": records(
            changed.sort_values("capacity_mw", ascending=False) if len(changed) else changed,
            ["site_name", "carrier", "capacity_mw", "support_before", "support_after",
             "banding_after", "units", "unit_names", "method", "note"]),
        "ro_unmatched_largest": records(
            unmatched.head(largest), ["unit_id", "Station name", "carrier", "capacity_mw", "Effective date"]),
        "cfd_notes": cfd_audit["note"].value_counts().to_dict() if len(cfd_audit) else {},
        "cfd_evidence_sites": records(
            cfd_audit[~cfd_audit["note"].eq("round confirmed")] if len(cfd_audit) else cfd_audit,
            ["site_name", "carrier", "capacity_mw", "units", "unit_names", "method",
             "round_repd", "round_lccc", "round_after", "cfd_capacity_after", "note"]),
    }


# Ofgem "Fuelled" stations that burn waste (the biomass fraction earns ROCs)
# belong to the CP30 waste_to_energy carrier; the rest (dedicated biomass,
# anaerobic digestion, landfill and sewage gas) to biomass.
_WASTE_STATION_RE = re.compile(
    r"efw|energy from waste|\berf\b|energy recovery|selchp|incinerat|recycling and renewable|"
    r"\bmkwrp\b|\brrc\b", re.IGNORECASE)


def register_proxy_carrier(station_name: object) -> str:
    """CP30 carrier of an Ofgem biomass-family station (waste_to_energy or biomass)."""
    return "waste_to_energy" if _WASTE_STATION_RE.search(str(station_name or "")) else "biomass"


def apply_ro_register_share_to_proxies(frame: pd.DataFrame, register: pd.DataFrame,
                                       carriers=("biomass", "waste_to_energy"),
                                       min_certificate_year: Optional[int] = 2023,
                                       carrier_col: str = "fuel_type",
                                       proxy_mask: Optional[pd.Series] = None,
                                       exclude_names: Iterable[str] = (),
                                       register_carrier: str = "biomass",
                                       classify=register_proxy_carrier,
                                       remainder_support: Optional[str] = None,
                                       logger: logging.Logger = LOGGER):
    """Give anonymous CP30 proxies the RO share the register implies.

    ``remainder_support`` relabels the part of each proxy the register does not
    hold as ROC (``None`` leaves it as it was). Marine uses ``"CfD"`` (item C4,
    27 Sep 2026): tidal stream built by 2030 beyond the RO fleet holds CfDs, the
    AR4 to AR7a ring-fenced awards, and none is viable merchant; the CfD vintage
    schedule then assigns its rounds.

    CP30 holds distribution biomass and energy from waste as FES spatial proxy
    rows with no names, so no station can be matched. The register still says
    how much of each carrier holds a live accreditation: GB stations eligible on
    the reference date that were issued certificates since
    ``min_certificate_year`` (the bill's activity rule, item MR8). Each proxy row
    of a carrier is split into a ROC part, ``share = min(1, register MW / proxy
    MW)`` of its capacity with the carrier's capacity-weighted observed banding,
    and the rest as it was. ``exclude_names`` removes register stations already
    placed by name (the biomass station schedule). CfD rows are never split.
    ``register_carrier`` selects the register's carrier (the bill's mapping:
    ``biomass`` for fuelled, landfill and sewage gas; ``marine`` for tidal and
    wave) and ``classify`` maps a station name to the proxy carrier.

    Returns (frame, audit by carrier).
    """
    if frame is None or frame.empty:
        return frame, pd.DataFrame()
    eligible = register[register["ro_eligible"] & register["gb"]
                        & register["carrier"].eq(register_carrier)].copy()
    if min_certificate_year is not None:
        eligible = eligible[pd.to_numeric(eligible["last_certificate_year"], errors="coerce")
                            .fillna(0).ge(int(min_certificate_year))]
    excluded_keys = [phase_free_key(name) for name in exclude_names]
    if excluded_keys and len(eligible):
        keys = eligible["Station name"].map(phase_free_key)
        placed = keys.map(lambda key: any(distinctive_overlap(key, other) for other in excluded_keys))
        eligible = eligible[~placed.astype(bool)]
    eligible["proxy_carrier"] = eligible["Station name"].map(classify)
    result = frame.copy()
    mask = (proxy_mask.reindex(result.index).fillna(False).astype(bool) if proxy_mask is not None
            else pd.Series(True, index=result.index))
    for column in ("ro_banding", "ro_support_source"):
        if column not in result.columns:
            result[column] = (np.nan if column == "ro_banding"
                              else pd.Series(pd.NA, index=result.index, dtype="object"))
    support = (result["support_type"] if "support_type" in result.columns
               else pd.Series("merchant", index=result.index))
    rows, audit, drop = [], [], []
    for carrier in carriers:
        stations = eligible[eligible["proxy_carrier"].eq(carrier)]
        target = float(stations["capacity_mw"].sum())
        members = result.index[mask & result[carrier_col].eq(carrier) & support.ne("CfD")]
        proxy_mw = float(pd.to_numeric(result.loc[members, "capacity_mw"], errors="coerce").sum())
        known = stations[stations["banding"].notna()]
        banding = (float((known["capacity_mw"] * known["banding"]).sum() / known["capacity_mw"].sum())
                   if len(known) and known["capacity_mw"].sum() > 0 else np.nan)
        share = min(1.0, target / proxy_mw) if proxy_mw > 0 else 0.0
        audit.append({"carrier": carrier, "register_active_mw": target, "register_stations": int(len(stations)),
                      "proxy_mw": proxy_mw, "roc_share": share, "roc_mw": share * proxy_mw,
                      "banding": banding, "unallocated_register_mw": max(target - proxy_mw, 0.0),
                      "remainder_support": remainder_support or "unchanged",
                      "remainder_mw": (1.0 - share) * proxy_mw})
        if share <= 0:
            if remainder_support:
                result.loc[members, "support_type"] = remainder_support
            continue
        for label in members:
            row = result.loc[label]
            capacity = float(row["capacity_mw"])
            roc = row.copy()
            roc["capacity_mw"] = capacity * share
            roc["support_type"] = "ROC"
            roc["ro_banding"] = round(banding, 4)
            roc["ro_support_source"] = "ofgem_ro_register_proxy_share"
            if share < 1.0:
                rest = row.copy()
                rest["capacity_mw"] = capacity * (1.0 - share)
                if remainder_support:
                    rest["support_type"] = remainder_support
                for name_column in ("site_name", "station_name"):
                    if name_column in row.index and pd.notna(row.get(name_column)):
                        roc[name_column] = f"{row[name_column]}_ROC"
                rows.append(rest)
            rows.append(roc)
            drop.append(label)
    if drop:
        result = pd.concat([result.drop(index=drop), pd.DataFrame(rows)], ignore_index=True, sort=False)
    audit = pd.DataFrame(audit)
    for entry in audit.itertuples(index=False):
        logger.info("Ofgem RO register, %s proxies: %.0f MW active accredited against %.0f MW proxy; "
                    "ROC share %.2f (%.0f MW, banding %.2f)", entry.carrier, entry.register_active_mw,
                    entry.proxy_mw, entry.roc_share, entry.roc_mw,
                    entry.banding if pd.notna(entry.banding) else 0.0)
        if remainder_support:
            logger.info("  %s proxies: the other %.0f MW labelled %s", entry.carrier, entry.remainder_mw,
                        remainder_support)
    return result, audit
