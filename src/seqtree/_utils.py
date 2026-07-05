from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def is_pandas_dataframe(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "to_dict") and hasattr(value, "__len__")


def normalize_table(data: Any) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Return records, column names, and whether the input looked DataFrame-like."""
    if is_pandas_dataframe(data):
        columns = [str(column) for column in data.columns]
        records = [{str(key): val for key, val in row.items()} for row in data.to_dict("records")]
        return records, columns, True

    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise TypeError("X must be a pandas DataFrame, a list of dictionaries, or a list of rows.")

    rows = list(data)
    if not rows:
        raise ValueError("X must contain at least one row.")

    first = rows[0]
    if isinstance(first, Mapping):
        columns = [str(column) for column in first.keys()]
        records = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("All rows must use the same representation.")
            records.append({str(key): value for key, value in row.items()})
        return records, columns, False

    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        width = len(first)
        columns = [f"x{i}" for i in range(width)]
        records = []
        for row in rows:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != width:
                raise TypeError("All rows must be sequences with the same length.")
            records.append({columns[i]: value for i, value in enumerate(row)})
        return records, columns, False

    raise TypeError("Rows must be dictionaries or sequences.")


def resolve_columns(columns: list[str], requested: Sequence[str | int] | None) -> list[str]:
    if requested is None:
        return list(columns)

    resolved = []
    for item in requested:
        if isinstance(item, int):
            try:
                resolved.append(columns[item])
            except IndexError as exc:
                raise ValueError(f"Column index out of range: {item}") from exc
        else:
            name = str(item)
            if name not in columns:
                raise ValueError(f"Unknown column in variable_order: {name!r}")
            resolved.append(name)

    if len(set(resolved)) != len(resolved):
        raise ValueError("variable_order cannot contain duplicate columns.")
    if set(resolved) != set(columns):
        missing = sorted(set(columns) - set(resolved))
        extra = sorted(set(resolved) - set(columns))
        raise ValueError(f"variable_order must include every column exactly once; missing={missing}, extra={extra}")
    return resolved


def resolve_column_subset(columns: list[str], requested: Sequence[str | int] | None, *, parameter: str) -> list[str]:
    if requested is None:
        return list(columns)

    resolved = []
    for item in requested:
        if isinstance(item, int):
            try:
                resolved.append(columns[item])
            except IndexError as exc:
                raise ValueError(f"Column index out of range in {parameter}: {item}") from exc
        else:
            name = str(item)
            if name not in columns:
                raise ValueError(f"Unknown column in {parameter}: {name!r}")
            resolved.append(name)

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{parameter} cannot contain duplicate columns.")
    return resolved


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_integer_code(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_float_value(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def require_numeric_values(records: list[dict[str, Any]], columns: list[str], *, backend: str) -> None:
    for column in columns:
        for row_index, record in enumerate(records):
            value = record.get(column)
            if value is None:
                continue
            if not is_number(value):
                raise TypeError(
                    f"tree_backend={backend!r} requires preprocessed numeric data; "
                    f"column {column!r} has non-numeric value {value!r} at row {row_index}."
                )


def validate_no_nulls(records: list[dict[str, Any]], columns: list[str]) -> None:
    for row_index, record in enumerate(records):
        for column in columns:
            if record.get(column) is None:
                raise ValueError(f"SeqTree does not accept null values; found null in column {column!r} at row {row_index}.")


def validate_variable_types(
    records: list[dict[str, Any]],
    *,
    continuous_columns: set[str],
    discrete_columns: set[str],
) -> None:
    for column in continuous_columns:
        for row_index, record in enumerate(records):
            value = record[column]
            if not is_float_value(value):
                raise TypeError(
                    f"continuous column {column!r} must contain float values; "
                    f"found {value!r} at row {row_index}."
                )

    for column in discrete_columns:
        for row_index, record in enumerate(records):
            value = record[column]
            if not is_integer_code(value):
                raise TypeError(
                    f"discrete column {column!r} must contain integer category codes; "
                    f"found {value!r} at row {row_index}."
                )
