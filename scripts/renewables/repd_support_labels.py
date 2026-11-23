"""REPD support labels: whitespace normalisation and sourced round corrections.

Two problems with the ``CfD Allocation Round`` and ``RO Banding`` fields as
REPD publishes them.

The first is cosmetic but leaks into reporting. Some rows carry a trailing
ASCII space or a non-breaking space (``"AR1 "`` beside ``"AR1"``). The strike
lookup strips before matching, so payments are unaffected, but every
``groupby("cfd_round")`` splits one round into two groups, which is how the
CfD vintage audit and the WS6 aggregation would double-count a round.

The second is substantive. REPD records an allocation round for two projects
whose support is not an allocation-round CfD at all. Hornsea Project One is
labelled ``AR3`` and Beatrice ``AR1``, but both hold Investment Contracts from
FID Enabling for Renewables, awarded in April 2014 and listed with their strike
prices in the NAO's *Early contracts for renewable electricity* (June 2014),
Figure 1. Neither appears in the AR1 or AR3 results: AR1 offshore was East
Anglia 1 and Neart na Gaoithe, and AR3 offshore was the Dogger Bank projects,
Sofia, Seagreen and Forthwind. Left uncorrected, Hornsea 1 settles 1,218 MW at
the AR3 offshore clearing price, roughly a third of its contracted strike.

Corrections are declared per project in a sourced CSV keyed by REPD Ref ID, not
by site name, because the names carry the whitespace this module also fixes.
"""

import pandas as pd

LABEL_COLUMNS = ("cfd_round", "ro_banding")

CORRECTION_COLUMNS = (
    "repd_ref_id", "site_name", "cfd_round_repd", "cfd_round", "source", "note",
)

# Unicode spaces seen in REPD text, mapped to a plain space before stripping.
_SPACE_CHARACTERS = {"\xa0": " ", " ": " ", " ": " ", " ": " "}


def normalise_label(values: pd.Series) -> pd.Series:
    """Trim stray unicode whitespace from a label column, preserving blanks."""
    text = values.astype("string")
    for character, replacement in _SPACE_CHARACTERS.items():
        text = text.str.replace(character, replacement, regex=False)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return text.mask(text.eq(""), pd.NA).where(values.notna(), values)


def normalise_support_labels(frame: pd.DataFrame, columns=LABEL_COLUMNS,
                             logger=None) -> pd.DataFrame:
    """Normalise the REPD support-label columns present on ``frame``."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        if not pd.api.types.is_string_dtype(result[column]) and \
                not pd.api.types.is_object_dtype(result[column]):
            continue  # numeric ro_banding needs no trimming
        before = result[column]
        after = normalise_label(before)
        changed = before.astype("string").ne(after) & before.notna()
        result[column] = after
        if logger is not None and int(changed.sum()):
            logger.info(
                "Normalised whitespace in %d %s label(s): %s",
                int(changed.sum()), column,
                sorted({repr(v) for v in before[changed].unique()})[:6],
            )
    return result


def load_cfd_round_corrections(path) -> pd.DataFrame:
    """Read the declared CfD round corrections, validating provenance."""
    frame = pd.read_csv(path, dtype=str).fillna("")
    if missing := set(CORRECTION_COLUMNS).difference(frame.columns):
        raise ValueError(f"CfD round corrections file is missing columns: {sorted(missing)}")
    frame["repd_ref_id"] = frame["repd_ref_id"].str.strip()
    for column in ("cfd_round", "source"):
        blank = frame[column].str.strip().eq("")
        if blank.any():
            raise ValueError(
                f"Every CfD round correction needs a {column}; "
                f"rows missing it: {sorted(frame.loc[blank, 'site_name'])}"
            )
    if frame["repd_ref_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame.repd_ref_id.duplicated(), "repd_ref_id"])
        raise ValueError(f"CfD round corrections have duplicate Ref IDs: {duplicates}")
    return frame


def apply_cfd_round_corrections(frame: pd.DataFrame, corrections: pd.DataFrame,
                                ref_id_column: str = "repd_ref_id",
                                logger=None) -> pd.DataFrame:
    """Replace ``cfd_round`` on the declared projects, matched on REPD Ref ID.

    A correction whose ``cfd_round_repd`` no longer matches what REPD publishes
    is an error: the register has been revised under the correction, and the
    evidence needs rechecking rather than silently overriding new data.
    """
    if corrections is None or corrections.empty or ref_id_column not in frame.columns:
        return frame
    result = frame.copy()
    if "cfd_round" not in result.columns:
        result["cfd_round"] = pd.NA
    # Ref IDs arrive as "2521", 2521 or 2521.0 depending on the read path.
    keys = (result[ref_id_column].astype("string").str.strip()
            .str.replace(r"\.0$", "", regex=True))
    applied = 0
    for row in corrections.itertuples(index=False):
        match = keys.eq(row.repd_ref_id)
        if not match.any():
            continue
        observed = normalise_label(result.loc[match, "cfd_round"]).fillna("")
        expected = str(row.cfd_round_repd).strip()
        if not observed.eq(expected).all():
            raise ValueError(
                f"REPD now records cfd_round {sorted(set(observed))} for Ref ID "
                f"{row.repd_ref_id} ({row.site_name}), but the correction expects "
                f"{expected!r}. Recheck the evidence before overriding."
            )
        result.loc[match, "cfd_round"] = row.cfd_round
        applied += int(match.sum())
        if logger is not None:
            logger.info(
                "CfD round correction: %s (Ref %s) %r -> %r [%s]",
                row.site_name, row.repd_ref_id, expected, row.cfd_round, row.source,
            )
    if logger is not None:
        logger.info("Applied %d of %d declared CfD round correction(s)",
                    applied, len(corrections))
    return result
