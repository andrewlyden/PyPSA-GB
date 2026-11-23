"""The model year of a finance and bill run, and the year-dependent names derived from it (item Y4).

Every finance and bill
script takes its year from the run it is accounting, never from a constant:

* a reference-run group's ``model_year`` (``public_ownership.reference_run_groups``);
* a forecast case's ``modelled_year`` (``consumer_bill_forecasts.cases``);
* or a file the chain already carries it in: the finance register's
  ``model_year`` column, the recovery-demand file's ``forecast_year``, the bill
  summary's ``modelled_year``.

:data:`DEFAULT_MODEL_YEAR` is used only where none of those is available (a
config group without the key, a test fixture, a CLI run without the flag), and
equals the CP30 2030 year so that every such default reproduces today's
behaviour.  It is the one place the year is written as a literal.
"""

import re
from typing import Iterable, Mapping

import pandas as pd

#: The year a run falls back to when neither its config nor its inputs name one (CP30 2030).
DEFAULT_MODEL_YEAR = 2030
#: From this model year a register is vintaged (FES 2050 plan, task Y10): every row
#: carries a build year, and is costed from the vintage table (Y2) by technology and
#: build year.  Below it the register takes ``technology_costs.csv`` as it always has,
#: so no 2030 number can move.
VINTAGED_REGISTER_FROM_MODEL_YEAR = 2040


def is_vintaged_model_year(model_year: int) -> bool:
    """Whether a register of this model year is vintaged (costed by build year, task Y10)."""
    return int(model_year) >= VINTAGED_REGISTER_FROM_MODEL_YEAR

#: Capacity Market rate columns the builder writes per delivery year and per calendar year
#: (``scripts/market/build_capacity_market.py``: ``revenue_2029_30_gbp``, ``calendar_2030_revenue_gbp``).
_CM_DELIVERY_YEAR_COLUMN = re.compile(r"^revenue_(\d{4})_(\d{2})_gbp$")
_CM_CALENDAR_COLUMN = re.compile(r"^calendar_(\d{4})_revenue_gbp$")
#: The per-kW rate, which is not year-labelled.
CM_RATE_PER_KW_COLUMN = "gbp_per_kw_connection_year"


def _year(value, what: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an integer year, got {value!r}")
    try:
        year = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be an integer year, got {value!r}") from exc
    if float(year) != float(value):
        raise ValueError(f"{what} must be an integer year, got {value!r}")
    return year


def group_model_year(group: Mapping | None) -> int:
    """A reference-run group's ``model_year``, else :data:`DEFAULT_MODEL_YEAR`.

    Args:
        group: one entry of ``public_ownership.reference_run_groups``.

    Returns:
        The group's model year.
    """
    value = (group or {}).get("model_year")
    return DEFAULT_MODEL_YEAR if value is None else _year(value, "reference_run_groups.<group>.model_year")


def recovery_model_year(recovery: Mapping | None) -> int:
    """The calendar year a recovery-demand file is for (its ``forecast_year``), else the default.

    Args:
        recovery: the parsed ``*_recovery_demand_assumptions.json``.

    Returns:
        The file's forecast year.
    """
    value = (recovery or {}).get("forecast_year")
    return DEFAULT_MODEL_YEAR if value is None else _year(value, "recovery demand forecast_year")


def frame_model_year(frame: pd.DataFrame, column: str = "model_year") -> int:
    """The one model year a register-like frame carries, refusing a frame that mixes years.

    Args:
        frame: a finance register (or any frame with a ``model_year`` column).
        column: the column name.

    Returns:
        The single model year; the default when the frame has no such column or no rows.
    """
    if column not in frame.columns or frame.empty:
        return DEFAULT_MODEL_YEAR
    years = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(years) == 0:
        return DEFAULT_MODEL_YEAR
    if len(years) != 1:
        raise ValueError(f"the frame mixes model years {sorted(years.tolist())}; one run group has one year")
    return _year(years[0], column)


def register_model_year(register_path) -> int:
    """The model year of a saved finance register (read from its private case only).

    Args:
        register_path: the ``*_asset_finance_register.parquet`` (or legacy CSV).

    Returns:
        The register's model year.
    """
    from scripts.public_ownership.register_io import read_register

    return frame_model_year(read_register(register_path, columns=["case_id", "model_year"], cases=["private"]))


def calendar_months(year: int) -> pd.PeriodIndex:
    """The twelve calendar months of ``year``, January first.

    Args:
        year: the model year.

    Returns:
        A monthly ``PeriodIndex``.
    """
    year = _year(year, "model year")
    return pd.period_range(f"{year}-01", f"{year}-12", freq="M")


def calendar_period(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The calendar year as ``[start, end)`` timestamps.

    Args:
        year: the model year.

    Returns:
        ``(1 January year, 1 January year + 1)``.
    """
    year = _year(year, "model year")
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")


def fiscal_year_label(start_year: int, separator: str = "_") -> str:
    """``2030`` -> ``2030_31``: the UK fiscal (and Capacity Market delivery) year starting 1 April.

    Args:
        start_year: the calendar year the fiscal year starts in.
        separator: between the two parts (``_`` for column names, ``/`` for text).

    Returns:
        The label.
    """
    start_year = _year(start_year, "fiscal year start")
    return f"{start_year}{separator}{(start_year + 1) % 100:02d}"


def cm_calendar_revenue_column(year: int) -> str:
    """The Capacity Market builder's calendar-year revenue column for ``year``.

    Args:
        year: the model year.

    Returns:
        ``calendar_<year>_revenue_gbp``.
    """
    return f"calendar_{_year(year, 'model year')}_revenue_gbp"


def cm_delivery_year_revenue_columns(year: int) -> tuple[str, str]:
    """The two delivery-year revenue columns a calendar year straddles (October-to-September years).

    Args:
        year: the model year.

    Returns:
        ``(revenue_<year-1>_<yy>_gbp, revenue_<year>_<yy+1>_gbp)``.
    """
    year = _year(year, "model year")
    return f"revenue_{fiscal_year_label(year - 1)}_gbp", f"revenue_{fiscal_year_label(year)}_gbp"


def is_cm_rate_column(name: str) -> bool:
    """Whether a column of ``cm_revenue_by_technology.csv`` is money (rebased with the register), for any year.

    Args:
        name: the column name.

    Returns:
        True for the per-kW rate, every delivery-year revenue and every calendar-year revenue.
    """
    return (name == CM_RATE_PER_KW_COLUMN or bool(_CM_DELIVERY_YEAR_COLUMN.match(name))
            or bool(_CM_CALENDAR_COLUMN.match(name)))


def cm_rate_columns(columns: Iterable[str]) -> list[str]:
    """The money columns of a CM rates table, in its own order.

    Args:
        columns: the table's columns.

    Returns:
        Those :func:`is_cm_rate_column` accepts.
    """
    return [c for c in columns if is_cm_rate_column(str(c))]


def year_offsets(model_year: int, offsets: Iterable[int]) -> tuple[int, ...]:
    """Years counted from the model year (the multi-year views: ``model_year + offset``).

    Args:
        model_year: the model year.
        offsets: non-negative offsets.

    Returns:
        The years, in the order given.
    """
    model_year = _year(model_year, "model year")
    return tuple(model_year + int(o) for o in offsets)
