"""The per-case asset finance register on disk: Parquet body plus a case-settings table (29 Sep 2026).

The register holds one row per asset per case (about 13,650 assets x 200 cases),
and 46 of its columns are the case's lever settings, identical on every row of a
case.  As CSV it was 884 MB at 60 cases.  It is now written as

* ``{run_group}_asset_finance_register.parquet``: the asset rows without the lever
  columns, zstd-compressed (Parquet dictionary-encodes the repeated text), and
* ``{run_group}_case_settings.csv``: one row per case, ``case_id`` plus every lever.

:func:`read_register` joins the two back.  Body columns come back exactly as the
CSV reader returned them (text as ``object``, missing values ``NaN``), so every
downstream number is unchanged; the lever columns come back as ``category``
(they are only ever compared or read per case, never grouped on), which is where
the memory went.  ``columns`` and ``cases`` restrict what is read.  A path ending
``.csv`` is read as before, so an old register still works.

30 Sep 2026, streaming: :class:`RegisterWriter` writes the register a block of
rows (one register key) at a time, and the file's metadata records the row
groups each case sits in (``case_row_groups``), so :func:`read_register` reads a
case from its own row groups; :func:`iter_register_cases` and :class:`CaseRows`
hand a stage a few cases at a time.  The data read back is unchanged; a file
written before the index is still read by filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.public_ownership.levers import ALL_LEVERS, LEVERS  # noqa: F401  (LEVERS kept for importers)

REGISTER_SUFFIX = "_asset_finance_register.parquet"
SETTINGS_SUFFIX = "_case_settings.csv"
META_KEY = b"pypsa_gb_register"
COMPRESSION = "zstd"


def lever_columns(columns: Iterable[str]) -> list[str]:
    """The lever columns among ``columns``: every catalogued lever, the year-gated ones included (FES 2050 plan)."""
    return [c for c in columns if c in ALL_LEVERS]


def lever_categorical(lever: str, values) -> pd.Categorical:
    """A lever column as a categorical over every value the lever can take, so any valid value can be set."""
    observed = [v for v in pd.unique(pd.Series(values, dtype=object).dropna())]
    categories = list(dict.fromkeys([*ALL_LEVERS[lever].values, *observed]))
    return pd.Categorical(values, categories=categories)


def settings_path(register_path: str | Path) -> Path:
    """``{run_group}_case_settings.csv`` beside ``{run_group}_asset_finance_register.parquet``."""
    path = Path(register_path)
    name = path.name
    if name.endswith(REGISTER_SUFFIX):
        return path.with_name(name[: -len(REGISTER_SUFFIX)] + SETTINGS_SUFFIX)
    return path.with_name(path.stem + SETTINGS_SUFFIX)


def _normalised_column(series: pd.Series) -> tuple[pd.Series, str]:
    """One object column as :func:`_normalise_for_parquet` writes it, and its kind.

    The per-value rule: ``None``, float NaN and the empty string are missing (the
    CSV writer left an empty string empty, and the reader gave it back missing);
    a column with nothing else is ``empty`` (float NaN); a column of bools is
    ``bool``, written ``"True"``/``"False"``; anything else is ``text``,
    written ``str(value)``.  30 Sep 2026: applied with array operations where
    the column holds only text, bools, ``None`` and float NaN (every register
    column), value by value otherwise, with the same result.
    """
    from pandas.api.types import infer_dtype

    array = series.to_numpy(dtype=object)
    try:
        empty_text = np.asarray(array == "", dtype=bool)
        is_none = np.asarray(array == None, dtype=bool)  # noqa: E711 (elementwise)
    except (TypeError, ValueError):
        return _normalised_column_by_value(series)
    missing_like = pd.isna(array)
    other_missing = missing_like & ~is_none
    if other_missing.any() and infer_dtype(array[other_missing], skipna=False) != "floating":
        return _normalised_column_by_value(series)       # pd.NA, NaT and the like: the per-value rule
    missing = missing_like | empty_text
    if missing.all():
        return pd.Series(np.nan, index=series.index, dtype="float64"), "empty"
    kind = infer_dtype(array[~missing], skipna=False)
    if kind == "boolean":
        out = np.where(missing, None, np.where(array.astype(bool), "True", "False")).astype(object)
        out[missing] = None
        return pd.Series(out, index=series.index, dtype=object), "bool"
    if kind == "string":
        out = array.copy()
        out[missing] = None
        return pd.Series(out, index=series.index, dtype=object), "text"
    return _normalised_column_by_value(series)


def _normalised_column_by_value(series: pd.Series) -> tuple[pd.Series, str]:
    """:func:`_normalised_column` value by value (the rule as first written, for unusual columns)."""
    series = series.map(lambda v: None if v is None or (isinstance(v, str) and v == "")
                        or (isinstance(v, float) and v != v) else v)
    values = series.dropna()
    if values.empty:
        return pd.Series(np.nan, index=series.index, dtype="float64"), "empty"
    if values.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        return series.map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(bool(v))), "bool"
    return series.map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v)), "text"


def _normalise_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    """Object columns as nullable text or numbers, the way the CSV reader typed them.

    A column mixing numbers and text (or bools and missing values) is written as
    text, and :func:`_restore` gives it back exactly as ``pandas.read_csv`` would.
    """
    out = frame.copy()
    kinds = {}
    for column in out.columns:
        series = out[column]
        if series.dtype != object and not isinstance(series.dtype, pd.CategoricalDtype):
            continue
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)
        out[column], kinds[column] = _normalised_column(series)
    out.attrs = {}
    return out, kinds


#: Rows per Parquet row group; each block of rows (one register key) starts a new group.
ROW_GROUP_ROWS = 200_000


def _part_kinds(body: pd.DataFrame, kinds: dict) -> dict[str, str]:
    """Each column's kind in one block: its object kind, else ``null`` (native but all missing) or ``native``."""
    out = {}
    for column in body.columns:
        if column in kinds:
            out[column] = kinds[column]
        else:
            out[column] = "null" if len(body) and body[column].isna().all() else "native"
    return out


def _unified(column: str, kinds: Sequence[str], types: Sequence) -> tuple[str | None, object]:
    """The kind and Arrow type one column takes over every block, as one frame of all the rows would give.

    The register used to be one frame, typed over all its rows, before it was
    written; written a block at a time, each block is typed alone, so the blocks
    are reconciled here.  A block where the column is wholly missing takes the
    others' type; text wins over bools; bools a block stored natively become
    text when another block held missing values (one frame of both is
    ``object``, which the normalisation writes as text); integers and floats
    meet as floats.  Anything else is refused.
    """
    import pyarrow as pa

    present = [(k, t) for k, t in zip(kinds, types) if k not in ("empty", "null")]
    if not present:
        return ("empty" if "empty" in kinds else None), pa.float64()
    numeric = [t for k, t in present if k == "native" and not pa.types.is_boolean(t)]
    if any(k == "text" for k, _ in present):
        if numeric:
            raise ValueError(f"register column {column!r} holds text in one block and numbers in another")
        return "text", pa.string()
    booleans = [t for k, t in present if k == "native" and pa.types.is_boolean(t)]
    if any(k == "bool" for k, _ in present) or (booleans and len(present) < len(kinds)):
        if numeric:
            raise ValueError(f"register column {column!r} holds bools in one block and numbers in another")
        return "bool", pa.string()
    native = {str(t) for _, t in present}
    if len(native) == 1:
        arrow = present[0][1]
        if len(present) < len(kinds) and pa.types.is_integer(arrow):
            return None, pa.float64()          # integers with missing blocks are floats in one frame
        return None, arrow
    if native <= {"int64", "double"}:
        return None, pa.float64()
    raise ValueError(f"register column {column!r} changes type across blocks: {sorted(native)}")


def _conform(table, schema):
    """One block's table in the register's unified schema."""
    import pyarrow as pa
    import pyarrow.compute as pc

    columns = []
    for field in schema:
        column = table[field.name]
        if column.type.equals(field.type):
            columns.append(column)
        elif column.null_count == len(column):
            columns.append(pa.nulls(len(column), field.type))
        elif pa.types.is_boolean(column.type) and pa.types.is_string(field.type):
            columns.append(pc.if_else(column, pa.scalar("True"), pa.scalar("False")))
        else:
            columns.append(column.cast(field.type))
    return pa.Table.from_arrays(columns, schema=schema)


def _pandas_metadata(columns: Sequence[str], types: Mapping[str, object]) -> bytes:
    """The ``pandas`` schema metadata one frame with these column types carries (the reader types by it)."""
    import pyarrow as pa

    sample = {}
    for column in columns:
        arrow = types[column]
        if pa.types.is_string(arrow):
            sample[column] = pd.Series(["x"], dtype=object)
        elif pa.types.is_boolean(arrow):
            sample[column] = pd.Series([True])
        else:
            sample[column] = pd.Series([1], dtype=arrow.to_pandas_dtype())
    return pa.Schema.from_pandas(pd.DataFrame(sample), preserve_index=False).metadata[b"pandas"]


def _distinct_settings(frame: pd.DataFrame, levers: Sequence[str]) -> pd.DataFrame:
    """``frame[["case_id", *levers]].drop_duplicates()``, read off each case's first row (F20 phase 1, 13 Oct 2026).

    A block carries one lever setting per case, so the distinct rows are each case's
    first row, in order; that is checked lever by lever (codes against the case's
    first row) and the whole-frame ``drop_duplicates`` runs only when it fails, so a
    case with two settings is still caught by :meth:`RegisterWriter.close`.
    """
    columns = ["case_id", *levers]
    codes, _ = pd.factorize(frame["case_id"], use_na_sentinel=False)
    first_rows = np.unique(codes, return_index=True)[1]            # each case's first row, cases in order
    for lever in levers:
        values = frame[lever]
        lever_codes = (values.cat.codes.to_numpy() if isinstance(values.dtype, pd.CategoricalDtype)
                       else pd.factorize(values, use_na_sentinel=False)[0])
        if not np.array_equal(lever_codes, lever_codes[first_rows][codes]):
            return frame[columns].drop_duplicates()
    return frame[columns].iloc[np.sort(first_rows)]


def _distinct_settings_reference(frame: pd.DataFrame, levers: Sequence[str]) -> pd.DataFrame:
    """The settings rows as the writer read them before F20 (the tests' reference)."""
    return frame[["case_id", *levers]].drop_duplicates()


class RegisterWriter:
    """Write the register a block of rows at a time, memory flat in the number of cases (30 Sep 2026).

    Each block (one register key's rows: every case sharing the key, lever
    columns attached) goes to a temporary Parquet part as it arrives.  Closing
    the writer reconciles the blocks' column types (:func:`_unified`) and writes
    the one register file, one or more row groups per block, with the row
    groups each case sits in recorded in the file's metadata
    (``case_row_groups``), so :func:`read_register` reads a case without
    scanning the file.  The file reads back exactly as :func:`write_register`
    of the concatenated blocks would.  A ``.csv`` path appends the blocks to
    one CSV instead.  Use it as a context manager, or call :meth:`close`.
    """

    def __init__(self, path: str | Path, row_group_rows: int = ROW_GROUP_ROWS):
        self.path = Path(path)
        self.row_group_rows = int(row_group_rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._columns: list[str] | None = None
        self._levers: list[str] = []
        self._parts: list[dict] = []
        self._settings: list[pd.DataFrame] = []
        self._scratch = self.path.with_name(f".{self.path.name}.parts")
        self._closed = False
        self.sizes: dict | None = None

    def __enter__(self) -> "RegisterWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._discard()

    def write(self, frame: pd.DataFrame) -> None:
        """Add one block of rows: whole cases, with the register's columns in its order."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._closed:
            raise ValueError("the register writer is closed")
        if self._columns is None:
            self._columns = list(frame.columns)
            self._levers = lever_columns(self._columns)
        elif list(frame.columns) != self._columns:
            raise ValueError("every block of the register must carry the same columns in the same order")
        if frame.empty:
            return
        if self.path.suffix == ".csv":
            frame.to_csv(self.path, index=False, mode="a" if self._parts else "w", header=not self._parts)
            self._parts.append({"rows": len(frame)})
            return
        self._settings.append(_distinct_settings(frame, self._levers).astype({c: object for c in self._levers}))
        body, kinds = _normalise_for_parquet(frame.drop(columns=self._levers))
        table = pa.Table.from_pandas(body, preserve_index=False)
        self._scratch.mkdir(parents=True, exist_ok=True)
        part = self._scratch / f"part_{len(self._parts):05d}.parquet"
        pq.write_table(table, part, compression=COMPRESSION)
        self._parts.append({"path": part, "rows": len(frame), "kinds": _part_kinds(body, kinds),
                            "types": {f.name: f.type for f in table.schema},
                            "case_id": frame["case_id"].astype(str).to_numpy()})

    def close(self) -> dict:
        """Write the register file and its case settings; return the file sizes."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._closed:
            return self.sizes
        self._closed = True
        if self.path.suffix == ".csv":
            self.sizes = {"register_bytes": self.path.stat().st_size if self.path.exists() else 0}
            return self.sizes
        if not self._parts:
            raise ValueError("the register writer was given no rows")
        settings = pd.concat(self._settings, ignore_index=True).drop_duplicates()
        if settings["case_id"].duplicated().any():
            repeated = sorted(settings.loc[settings["case_id"].duplicated(), "case_id"].astype(str).unique())[:5]
            self._discard()
            raise ValueError(f"the register carries more than one lever setting for cases {repeated}")
        settings.to_csv(settings_path(self.path), index=False)
        body_columns = [c for c in self._columns if c not in self._levers]
        kinds, types = {}, {}
        for column in body_columns:
            kind, types[column] = _unified(column, [p["kinds"][column] for p in self._parts],
                                           [p["types"][column] for p in self._parts])
            if kind is not None:
                kinds[column] = kind
        # Row groups: consecutive blocks are pooled up to ``row_group_rows`` (a block
        # larger than that is split), so the groups are as large as one frame's
        # were and each block's cases sit in as few groups as possible.
        pools: list[list[dict]] = []
        for part in self._parts:
            if pools and sum(p["rows"] for p in pools[-1]) + part["rows"] <= self.row_group_rows:
                pools[-1].append(part)
            else:
                pools.append([part])
        case_groups: dict[str, list[int]] = {}
        group = 0
        for pool in pools:
            case_ids = np.concatenate([p["case_id"] for p in pool])
            for start in range(0, len(case_ids), self.row_group_rows):
                for case in pd.unique(case_ids[start:start + self.row_group_rows]):
                    case_groups.setdefault(str(case), []).append(group)
                group += 1
        schema = pa.schema([pa.field(c, types[c]) for c in body_columns])
        meta = {b"pandas": _pandas_metadata(body_columns, types),
                META_KEY: json.dumps({"column_order": self._columns, "lever_columns": self._levers,
                                      "object_kinds": kinds, "case_row_groups": case_groups}).encode("utf-8")}
        with pq.ParquetWriter(self.path, schema.with_metadata(meta), compression=COMPRESSION) as writer:
            for pool in pools:
                table = pa.concat_tables([_conform(pq.read_table(p["path"]), schema) for p in pool])
                writer.write_table(table, row_group_size=self.row_group_rows)
                del table
        self._discard()
        if pq.ParquetFile(self.path).num_row_groups != group:
            raise RuntimeError("the register's row groups do not match its case index")
        self.sizes = {"register_bytes": self.path.stat().st_size,
                      "settings_bytes": settings_path(self.path).stat().st_size}
        return self.sizes

    def _discard(self) -> None:
        import shutil

        shutil.rmtree(self._scratch, ignore_errors=True)


def write_register(register: pd.DataFrame, path: str | Path) -> dict:
    """Write the body as Parquet and the per-case lever settings as CSV; return file sizes.

    The whole register as one block of :class:`RegisterWriter` (one row group
    per 200,000 rows, as before, now with the case index).
    """
    with RegisterWriter(path) as writer:
        writer.write(register)
    return writer.sizes


def _restore(frame: pd.DataFrame, kinds: dict) -> pd.DataFrame:
    """Give text back as ``object`` with ``NaN`` for missing (the CSV reader's typing), bools as bools.

    F20 phase 1 (13 Oct 2026): the same result as :func:`_restore_reference`, with array
    operations.  A text column is tested for being all numbers on its distinct values (a
    value parses or not whatever its neighbours), and only a column that is all numbers is
    parsed whole, exactly as before; a bool column is mapped as arrays where it holds only
    ``"True"``/``"False"`` text and missing values, value by value otherwise.  On a row
    group of the 2030 register this took the restore from about 4.6 s to well under one.
    """
    for column, kind in kinds.items():
        if column not in frame.columns:
            continue
        series = frame[column]
        if kind == "empty":
            frame[column] = series.astype("float64")
        elif kind == "bool":
            frame[column] = _restored_bool(series)
            if frame[column].notna().all():
                frame[column] = frame[column].astype(bool)
        else:
            present = series.notna()          # = values.notna(): only missing values become NaN
            values = series.astype(object).where(present, np.nan)
            # The CSV reader parsed an all-numeric text column as numbers.
            if present.sum() and _all_numeric(values[present]):
                frame[column] = pd.to_numeric(values, errors="coerce")
            else:
                frame[column] = values
    return frame


def _all_numeric(values: pd.Series) -> bool:
    """Whether every one of these (non-missing) values parses as a number (``pd.to_numeric``), tested on distinct values."""
    distinct = pd.Series(pd.unique(values.to_numpy(dtype=object)), dtype=object)
    return bool(pd.to_numeric(distinct, errors="coerce").notna().all())


def _restored_bool(series: pd.Series) -> pd.Series:
    """A bool column's text as Python bools, ``NaN`` for missing: :func:`_restore_reference`'s map as arrays."""
    from pandas._libs import lib

    array = series.to_numpy(dtype=object)
    missing = pd.isna(array)
    text = array[~missing]
    plain = ((not len(text) or lib.is_string_array(text, skipna=False))
             and all(v is None or isinstance(v, float) for v in array[missing]))
    if not plain:
        return series.map(lambda v: np.nan if v is None or (isinstance(v, float) and v != v)
                          else v == "True").astype(object)
    out = np.empty(len(array), dtype=object)
    out[missing] = np.nan
    out[~missing] = array[~missing] == "True"
    return pd.Series(out, index=series.index, dtype=object)


def _restore_reference(frame: pd.DataFrame, kinds: dict) -> pd.DataFrame:
    """:func:`_restore` as first written, value by value (kept as the reference the tests check it against)."""
    for column, kind in kinds.items():
        if column not in frame.columns:
            continue
        series = frame[column]
        if kind == "empty":
            frame[column] = series.astype("float64")
        elif kind == "bool":
            frame[column] = series.map(lambda v: np.nan if v is None or (isinstance(v, float) and v != v)
                                       else v == "True").astype(object)
            if frame[column].notna().all():
                frame[column] = frame[column].astype(bool)
        else:
            values = series.astype(object).where(series.notna(), np.nan)
            # The CSV reader parsed an all-numeric text column as numbers.
            numeric = pd.to_numeric(values, errors="coerce")
            if values.notna().sum() and numeric.notna().sum() == values.notna().sum():
                frame[column] = numeric
            else:
                frame[column] = values
    return frame


def _parquet_plan(path: Path, columns: Sequence[str] | None, wanted_cases) -> dict:
    """What a Parquet read needs: the metadata, the columns to read and the order to return them in."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    meta = json.loads((schema.metadata or {}).get(META_KEY, b"{}"))
    order = meta.get("column_order") or list(schema.names)
    levers = meta.get("lever_columns") or []
    requested = list(order) if columns is None else [c for c in columns if c in order]
    body_cols = [c for c in requested if c in schema.names]
    lever_cols = [c for c in requested if c in levers]
    read_cols = list(dict.fromkeys((["case_id"] if (lever_cols or wanted_cases is not None) else []) + body_cols))
    keep = set(requested) | ({"case_id"} if wanted_cases is not None else set())
    return {"meta": meta, "read_cols": read_cols, "lever_cols": lever_cols,
            "returned": [c for c in order if c in keep], "index": meta.get("case_row_groups")}


def _frame_from_table(path: Path, table, plan: Mapping, lever_dtype: str, settings: pd.DataFrame | None = None):
    """A read Arrow table as the register frame: text restored as the CSV reader typed it, levers attached."""
    frame = _restore(table.to_pandas(), plan["meta"].get("object_kinds") or {})
    if plan["lever_cols"]:
        if settings is None:
            settings = pd.read_csv(settings_path(path), dtype=str, keep_default_na=False, na_values=[""])
            settings = settings.set_index("case_id")
        _attach_levers(frame, settings, plan["lever_cols"], lever_dtype)
    return frame[plan["returned"]]


def _attach_levers(frame: pd.DataFrame, settings: pd.DataFrame, levers: Sequence[str], lever_dtype: str) -> None:
    """Each lever column from the settings table, by case (in place).

    F20 phase 1 (13 Oct 2026): every row of a case carries the case's settings, so each
    lever is looked up once per case and spread over the rows by the case's position; the
    columns equal :func:`_attach_levers_reference`'s (the categories in the order the
    values first appear, as there).
    """
    codes, cases = pd.factorize(frame["case_id"].astype(str))
    by_case = pd.Series(cases, dtype=object)
    for lever in levers:
        values = by_case.map(settings[lever])
        if lever_dtype == "category":
            per_case = lever_categorical(lever, values)
            frame[lever] = pd.Categorical.from_codes(per_case.codes[codes], dtype=per_case.dtype)
        else:
            frame[lever] = pd.Series(values.to_numpy(dtype=object)[codes], index=frame.index, dtype=object)


def _attach_levers_reference(frame: pd.DataFrame, settings: pd.DataFrame, levers: Sequence[str],
                             lever_dtype: str) -> None:
    """:func:`_attach_levers` as first written, row by row (kept as the reference the tests check it against)."""
    for lever in levers:
        values = frame["case_id"].astype(str).map(settings[lever])
        frame[lever] = lever_categorical(lever, values) if lever_dtype == "category" else values.astype(object)


def _keep_cases(table, cases: Sequence[str]):
    import pyarrow as pa
    import pyarrow.compute as pc

    return table.filter(pc.is_in(table["case_id"], value_set=pa.array(list(cases), type=pa.string())))


def _read_case_groups(path: Path, read_cols: Sequence[str], cases: Sequence[str], index: Mapping):
    """The rows of ``cases`` from only the row groups the case index names, in file order."""
    import pyarrow.parquet as pq

    import pyarrow as pa

    groups = sorted({g for case in cases for g in index.get(case, [])})
    handle = pq.ParquetFile(path)
    if not groups:
        return handle.schema_arrow.empty_table().select(list(read_cols))
    # One row group at a time, kept rows only: a few cases from many groups never hold every group at once.
    return pa.concat_tables([_keep_cases(handle.read_row_group(g, columns=list(read_cols)), cases) for g in groups])


def read_register(path: str | Path, columns: Sequence[str] | None = None, cases: Iterable[str] | None = None,
                  lever_dtype: str = "category") -> pd.DataFrame:
    """The register as the CSV reader gave it (lever columns ``category``), optionally a column/case subset.

    A file written with a case index (30 Sep 2026) reads only the row groups
    holding ``cases``; an older file is filtered as before.
    """
    path = Path(path)
    wanted_cases = None if cases is None else sorted({str(c) for c in cases})
    if path.suffix == ".csv":
        header = pd.read_csv(path, nrows=0).columns
        use = [c for c in (columns or header) if c in header]
        if "case_id" not in use and wanted_cases is not None:
            use = ["case_id", *use]
        parts = [chunk if wanted_cases is None else chunk.loc[chunk["case_id"].isin(wanted_cases)]
                 for chunk in pd.read_csv(path, usecols=use, chunksize=500_000, low_memory=False)]
        return pd.concat(parts, ignore_index=True)
    import pyarrow.parquet as pq

    plan = _parquet_plan(path, columns, wanted_cases)
    if wanted_cases is not None and plan["index"] is not None:
        table = _read_case_groups(path, plan["read_cols"], wanted_cases, plan["index"])
    else:
        filters = [("case_id", "in", wanted_cases)] if wanted_cases is not None else None
        table = pq.read_table(path, columns=plan["read_cols"], filters=filters)
    return _frame_from_table(path, table, plan, lever_dtype)


class RegisterReader:
    """:func:`read_register` for a stage that reads the same few cases with every batch (F20 phase 1, 13 Oct 2026).

    The tariff reads private with each batch of tariff cases; a read here returns
    exactly what ``read_register(path, columns, cases)`` returns, but the settings
    table is read once and the row groups holding only ``pinned`` cases are read
    once and kept (filtered to those cases), so private's groups are not decoded
    again for every batch.  A group a batch shares with a pinned case is read as
    before.  A file without the case index (or a CSV) is read by :func:`read_register`.
    """

    def __init__(self, path: str | Path, columns: Sequence[str] | None = None, pinned: Iterable[str] = ("private",),
                 lever_dtype: str = "category"):
        self.path = Path(path)
        self.columns = None if columns is None else list(columns)
        self.lever_dtype = lever_dtype
        self.pinned = {str(c) for c in pinned}
        self._index = None
        self._settings = None
        self._pinned_tables: dict[int, object] = {}
        if self.path.suffix != ".csv":
            plan = _parquet_plan(self.path, self.columns, sorted(self.pinned))
            self._index = plan["index"]
        if self._index is not None:
            self._cases_in_group: dict[int, set] = {}
            for case, groups in self._index.items():
                for g in groups:
                    self._cases_in_group.setdefault(g, set()).add(case)

    def read(self, cases: Iterable[str]) -> pd.DataFrame:
        """The rows of ``cases``, as ``read_register(path, columns, cases, lever_dtype)`` returns them."""
        wanted = sorted({str(c) for c in cases})
        if self._index is None:
            return read_register(self.path, columns=self.columns, cases=wanted, lever_dtype=self.lever_dtype)
        import pyarrow as pa
        import pyarrow.parquet as pq

        plan = _parquet_plan(self.path, self.columns, wanted)
        groups = sorted({g for case in wanted for g in self._index.get(case, [])})
        handle = pq.ParquetFile(self.path)
        if not groups:
            table = handle.schema_arrow.empty_table().select(list(plan["read_cols"]))
        else:
            parts = []
            wanted_set = set(wanted)
            for g in groups:
                needed = self._cases_in_group.get(g, set()) & wanted_set
                if needed and needed <= self.pinned:
                    # A group only pinned cases are read from: their rows once, then filtered to those asked for.
                    if g not in self._pinned_tables:
                        held = self._cases_in_group.get(g, set()) & self.pinned
                        self._pinned_tables[g] = _keep_cases(handle.read_row_group(g, columns=list(plan["read_cols"])),
                                                             sorted(held))
                    part = self._pinned_tables[g]
                    if needed != self._cases_in_group.get(g, set()) & self.pinned:
                        part = _keep_cases(part, wanted)
                    parts.append(part)
                else:
                    parts.append(_keep_cases(handle.read_row_group(g, columns=list(plan["read_cols"])), wanted))
            table = pa.concat_tables(parts)
        if plan["lever_cols"] and self._settings is None:
            self._settings = pd.read_csv(settings_path(self.path), dtype=str, keep_default_na=False,
                                         na_values=[""]).set_index("case_id")
        return _frame_from_table(self.path, table, plan, self.lever_dtype, self._settings)


def frames_identical(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Whether two frames hold the same columns, dtypes and values, bit for bit (the index is not compared).

    F20 phase 1 (13 Oct 2026): the test a stage uses before reusing a result computed on
    other rows.  Floats are compared as bits (so ``-0.0`` is not ``0.0``; a NaN equals a
    NaN with the same bits), categoricals by categories in order and codes, text by value
    and missing values by type (``None`` is not ``NaN``).  A dtype it does not know is
    never identical.
    """
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    for i in range(a.shape[1]):
        x, y = a.iloc[:, i], b.iloc[:, i]
        if x.dtype != y.dtype:
            return False
        if isinstance(x.dtype, pd.CategoricalDtype):
            if not (x.cat.categories.equals(y.cat.categories) and x.cat.ordered == y.cat.ordered
                    and x.cat.categories.dtype == y.cat.categories.dtype
                    and np.array_equal(x.cat.codes.to_numpy(), y.cat.codes.to_numpy())):
                return False
        elif not isinstance(x.dtype, np.dtype):
            return False
        elif x.dtype.kind == "f":
            bits = np.dtype(f"i{x.dtype.itemsize}")
            if not np.array_equal(x.to_numpy().view(bits), y.to_numpy().view(bits)):
                return False
        elif x.dtype.kind in "biu":
            if not np.array_equal(x.to_numpy(), y.to_numpy()):
                return False
        elif x.dtype == object:
            if not _objects_identical(x.to_numpy(), y.to_numpy()):
                return False
        else:
            return False
    return True


def _objects_identical(u: np.ndarray, v: np.ndarray) -> bool:
    """Two object arrays element by element: the same type and value (floats as bits, any NaN alike)."""
    from pandas._libs import lib

    missing_u, missing_v = pd.isna(u), pd.isna(v)
    if not np.array_equal(missing_u, missing_v):
        return False
    if [type(e) for e in u[missing_u]] != [type(e) for e in v[missing_v]]:
        return False
    text_u, text_v = u[~missing_u], v[~missing_v]
    if lib.is_string_array(text_u, skipna=False) and lib.is_string_array(text_v, skipna=False):
        return bool(np.array_equal(text_u, text_v))
    import struct

    for p, q in zip(text_u, text_v):
        if type(p) is not type(q):
            return False
        if isinstance(p, float):
            if struct.pack("<d", p) != struct.pack("<d", q):
                return False
            continue
        try:
            if not bool(p == q):
                return False
        except Exception:  # noqa: BLE001  (a value that cannot be compared is never identical)
            return False
    return True


def register_cases(path: str | Path) -> list[str]:
    """Every case in the register, in register order."""
    path = Path(path)
    if settings_path(path).exists() and path.suffix != ".csv":
        return read_settings(path)["case_id"].astype(str).tolist()
    ids = read_register(path, columns=["case_id"])["case_id"].astype(str)
    return list(dict.fromkeys(ids))


def iter_register_cases(path: str | Path, cases: Iterable[str] | None = None, columns: Sequence[str] | None = None,
                        batch_cases: int | None = 8, lever_dtype: str = "category") -> Iterator[pd.DataFrame]:
    """The register a few whole cases at a time, in register order (30 Sep 2026).

    Each frame is what :func:`read_register` returns for its cases (at most
    ``batch_cases``), so a stage that works case by case holds one batch, not
    the register.  With the case index, cases that share row groups (one
    register key) are read from one read of those groups.  ``batch_cases=None``
    yields each such run whole: whole register keys, in register order, so a
    stage whose output order follows the rows (a ``groupby(sort=False)`` across
    cases) gets the same order batch by batch (a file without the index is
    then read whole).
    """
    path = Path(path)
    wanted = None if cases is None else {str(c) for c in cases}
    order = [c for c in register_cases(path) if wanted is None or c in wanted]
    if not order:
        return
    plan = _parquet_plan(path, columns, order) if path.suffix != ".csv" else None
    if batch_cases is None and (plan is None or plan["index"] is None):
        yield read_register(path, columns=columns, cases=order, lever_dtype=lever_dtype)
        return
    if plan is None or plan["index"] is None:
        batch_cases = max(1, int(batch_cases))
        for start in range(0, len(order), batch_cases):
            yield read_register(path, columns=columns, cases=order[start:start + batch_cases], lever_dtype=lever_dtype)
        return
    import pyarrow.parquet as pq

    index = plan["index"]
    handle = pq.ParquetFile(path)
    settings = None
    if plan["lever_cols"]:
        settings = pd.read_csv(settings_path(path), dtype=str, keep_default_na=False, na_values=[""]).set_index("case_id")
    # Runs of consecutive cases that sit in the same row groups (one register key's cases).
    runs: list[tuple[tuple[int, ...], list[str]]] = []
    for case in order:
        groups = tuple(index.get(case, ()))
        if runs and runs[-1][0] == groups:
            runs[-1][1].append(case)
        else:
            runs.append((groups, [case]))
    for groups, run in runs:
        table = (handle.read_row_groups(list(groups), columns=plan["read_cols"]) if groups
                 else handle.schema_arrow.empty_table().select(plan["read_cols"]))
        size = len(run) if batch_cases is None else max(1, int(batch_cases))
        for start in range(0, len(run), size):
            batch = run[start:start + size]
            yield _frame_from_table(path, _keep_cases(table, batch), plan, lever_dtype, settings)
        del table


class CaseRows:
    """A register's rows a case at a time, from a file (read on demand) or from a frame (30 Sep 2026).

    For a stage that looks cases up one by one (``register.loc[register["case_id"].eq(case)]``)
    over every case: :meth:`rows` returns the rows of the cases asked for, each
    case's rows exactly as that filter over the whole register returned them.
    From a file only the cases in use are held: a case's row groups are read
    once, every case in them is kept, and at most ``cache_cases`` cases stay
    in memory (private is always kept), so memory is flat in the number of cases.

    ``share_key`` (F20 phase 2, 27 Oct 2026): ``case_id -> key``, equal for cases whose rows
    are the same apart from their labels (``fingerprints.body_key``).  :meth:`memo` and
    :meth:`prefetch` then compute a request once per label and keys of its cases, and give
    every case on the same keys a copy, so only the first case of each key is read; the
    caller passes it only for requests that read no label (not ``case_id`` beyond selecting
    the rows, no lever column outside the key).  A sample of the shared requests is computed
    in full as well and must match (``fingerprints.SampleCheck``).
    """

    def __init__(self, source, columns: Sequence[str] | None = None, cases: Iterable[str] | None = None,
                 cache_cases: int = 48, share_key=None):
        from collections import OrderedDict

        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._cache_cases = max(1, int(cache_cases))
        self._memo: dict = {}
        self._memo_cases: dict = {}
        self._share_key = share_key
        self.sample = None
        self._reader = None
        wanted = None if cases is None else {str(c) for c in cases}
        if isinstance(source, pd.DataFrame):
            self._frame, self.path = source, None
            self.columns = source.columns
            self.cases = set(source["case_id"]) if "case_id" in source.columns else set()
            self.order = [str(c) for c in dict.fromkeys(source["case_id"])] if "case_id" in source.columns else []
            return
        self._frame, self.path = None, Path(source)
        order = [c for c in register_cases(self.path) if wanted is None or c in wanted]
        self.cases = set(order)
        self.order = order
        if self.path.suffix == ".csv":
            header = pd.read_csv(self.path, nrows=0).columns
            self._columns = [c for c in (columns or header) if c in header]
            self.columns = pd.Index(list(dict.fromkeys(["case_id", *self._columns])))
            self._runs = {c: (c,) for c in order}
            return
        plan = _parquet_plan(self.path, columns, order)
        self._columns = list(columns) if columns is not None else None
        self.columns = pd.Index(plan["returned"])
        index = plan["index"] or {}
        by_groups: dict[tuple, list[str]] = {}
        for case in order:
            by_groups.setdefault(tuple(index.get(case, ())), []).append(case)
        # The cases read together with each case: those in the same row groups (none without an index).
        self._runs = {case: (tuple(run) if index else (case,)) for run in by_groups.values() for case in run}
        # F20 phase 1: reads go through one reader, so the settings table is read once, not with every run.
        self._reader = RegisterReader(self.path, columns=self._columns, pinned=())

    @classmethod
    def of(cls, source) -> "CaseRows":
        """``source`` as a :class:`CaseRows` (a frame is wrapped, a CaseRows passed through)."""
        return source if isinstance(source, CaseRows) else cls(source)

    def rows(self, *case_ids: str) -> pd.DataFrame:
        """The rows of these cases (each case's rows in register order, the cases in the order asked)."""
        if self._frame is not None:
            frame = self._frame
            return frame.loc[frame["case_id"].isin([str(c) for c in case_ids])]
        parts = [self._case(str(case_id)) for case_id in dict.fromkeys(case_ids)]
        return parts[0] if len(parts) == 1 else pd.concat(parts)

    def memo(self, label: str, case_ids: Sequence[str], compute):
        """``compute(self.rows(*case_ids))``, computed once per ``(label, case_ids)`` (F20 phase 1, 13 Oct 2026).

        For a stage that passes over every case several times, each pass deriving a
        small quantity from a case's rows: the quantity is kept (not the rows), and
        each caller gets its own deep copy.  ``label`` must name the computation and
        every constant it depends on beyond the rows; :meth:`clear_memo` forgets all.
        """
        import copy

        key = self._memo_key(label, case_ids)
        if key not in self._memo:
            self._memo[key] = compute(self.rows(*case_ids))
            self._memo_cases[key] = tuple(str(c) for c in case_ids)
        elif self._memo_cases.get(key) != tuple(str(c) for c in case_ids) and self._sample().due():
            # F20 phase 2: a result shared from other cases on the same keys, recomputed in full now and then.
            import json

            same = (json.dumps(compute(self.rows(*case_ids)), sort_keys=True, default=repr)
                    == json.dumps(self._memo[key], sort_keys=True, default=repr))
            self.sample.check(f"{label} of {', '.join(map(str, case_ids))}", same)
        return copy.deepcopy(self._memo[key])

    def _memo_key(self, label: str, case_ids: Sequence[str]) -> tuple:
        """A request's memo key: its label and case ids, or with ``share_key`` the keys of its cases."""
        if self._share_key is None:
            return (label, tuple(str(c) for c in case_ids))
        return (label, tuple(self._share_key(str(c)) for c in case_ids))

    def _sample(self):
        if self.sample is None:
            from scripts.public_ownership.fingerprints import SampleCheck

            self.sample = SampleCheck("register_io.CaseRows.memo")
        return self.sample

    def prefetch(self, requests: Iterable[tuple]) -> int:
        """Compute ``(label, case_ids, compute)`` requests for :meth:`memo` in register order; return how many.

        Walking the cases in register order reads each run of row groups once, where
        passes in their own order read them again each time (the cache holds
        ``cache_cases`` cases).  A request for a case the register does not hold, or
        whose computation raises, is left alone: :meth:`memo` computes it when asked,
        and raises there as the stage always did.
        """
        position = {case: i for i, case in enumerate(self.order)}
        done = 0
        for label, case_ids, compute in sorted(requests, key=lambda r: position.get(str(r[1][0]), len(position))):
            if any(str(c) not in self.cases for c in case_ids):
                continue
            key = self._memo_key(label, case_ids)
            if key in self._memo:
                continue
            try:
                self._memo[key] = compute(self.rows(*case_ids))
            except Exception:  # noqa: BLE001  (raised again, in place, by the stage that asks for it)
                continue
            self._memo_cases[key] = tuple(str(c) for c in case_ids)
            done += 1
        return done

    def clear_memo(self) -> None:
        """Forget every :meth:`memo` result."""
        self._memo.clear()
        self._memo_cases.clear()

    def _case(self, case_id: str) -> pd.DataFrame:
        if case_id in self._cache:
            self._cache.move_to_end(case_id)
            return self._cache[case_id]
        run = self._runs.get(case_id, (case_id,))
        frame = (self._reader.read(list(run)) if self._reader is not None
                 else read_register(self.path, columns=self._columns, cases=list(run)))
        found = {}
        # F20 phase 1: the run's rows split by case in one pass (each case's rows as the
        # ``frame.loc[frame["case_id"].eq(case)]`` they replace: the same rows, index and order).
        positions = frame.groupby("case_id", sort=False).indices if len(run) > 1 else {}
        none = np.array([], dtype=np.intp)
        for case in run:
            found[case] = frame.iloc[positions.get(case, none)] if len(run) > 1 else frame
        del frame
        for case, rows in found.items():
            self._cache[case] = rows
            self._cache.move_to_end(case)
        while len(self._cache) > self._cache_cases + len(run):
            oldest = next((c for c in self._cache if c != "private" and c not in found), None)
            if oldest is None:
                break
            self._cache.pop(oldest)
        return self._cache[case_id]


def read_settings(path: str | Path) -> pd.DataFrame:
    """The case-settings table of a register (``case_id`` plus every lever)."""
    return pd.read_csv(settings_path(path), dtype=str, keep_default_na=False, na_values=[""])


def market_bill_cases(path: str | Path) -> list[str] | None:
    """The register's cases that go through the market bill, from its settings table; None for an old CSV.

    The market-bill stages (CfD re-pricing, the public account) skip every case that
    acquires the legacy fleet (``spec_for_row(...).market_bill()``), so they need only
    these rows: at 210 cases that is 127, not 210.
    """
    from scripts.public_ownership.levers import spec_for_row

    if not settings_path(path).exists():
        return None
    table = read_settings(path)
    return [str(r["case_id"]) for r in table.to_dict("records")
            if spec_for_row({k: v for k, v in r.items() if k != "case_id"}).market_bill()]
