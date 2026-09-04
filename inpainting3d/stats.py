from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import stats


RESERVED_COLUMNS = {
    "index",
    "case_id",
    "path",
    "split",
    "experiment",
    "label",
    "model",
    "objective",
    "method",
    "train_seed",
    "eval_seed",
    "source_eval_dir",
    "weight_path",
    "slice_geometry",
}


def metric_columns_from_rows(rows: Sequence[Mapping[str, object]]) -> List[str]:
    """
    Extract metric column names from case-level metric rows.

    Parameters
    ----------
    rows : Sequence[Mapping[str, object]]
        Row dictionaries that should be written or summarized.

    Returns
    -------
    List[str]
        Metric column names extracted from the provided rows.
    """
    columns: set[str] = set()
    for row in rows:
        columns.update(str(key) for key in row.keys())
    metric_columns = [name for name in sorted(columns) if name not in RESERVED_COLUMNS]
    return metric_columns


def summarize_metric_values(values: Iterable[float], confidence: float = 0.95) -> Dict[str, float | int]:
    """
    Summarize a metric sequence with descriptive statistics and confidence intervals.

    Parameters
    ----------
    values : Iterable[float]
        Numeric values that should be summarized.
    confidence : float
        Confidence level used for interval estimation. Defaults to 0.95.

    Returns
    -------
    Dict[str, float | int]
        Computed summary values.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty metric sequence.")

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr = float(q75 - q25)
    if arr.size > 1:
        std = float(np.std(arr, ddof=1))
        sem = stats.sem(arr)
        alpha = 0.5 * (1.0 + confidence)
        delta = float(stats.t.ppf(alpha, df=arr.size - 1) * sem)
        ci_low = mean - delta
        ci_high = mean + delta
    else:
        std = 0.0
        ci_low = mean
        ci_high = mean

    return {
        "n": int(arr.size),
        "mean": mean,
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "std": std,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def summarize_case_metrics(
    rows: Sequence[Mapping[str, object]],
    metric_columns: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float | int]]:
    """
    Summarize case-level metrics across all selected metric columns.

    Parameters
    ----------
    rows : Sequence[Mapping[str, object]]
        Row dictionaries that should be written or summarized.
    metric_columns : Sequence[str] | None
        Metric names that should be summarized. Defaults to None.
    confidence : float
        Confidence level used for interval estimation. Defaults to 0.95.

    Returns
    -------
    Dict[str, Dict[str, float | int]]
        Computed summary values.
    """
    if not rows:
        raise ValueError("Cannot summarize metrics without case-level rows.")

    metrics = list(metric_columns) if metric_columns is not None else metric_columns_from_rows(rows)
    summary: Dict[str, Dict[str, float | int]] = {}
    for metric_name in metrics:
        values = [float(row[metric_name]) for row in rows if metric_name in row]
        if not values:
            continue
        summary[metric_name] = summarize_metric_values(values, confidence=confidence)
    return summary


def summary_rows_from_nested(summary: Mapping[str, Mapping[str, float | int]]) -> List[Dict[str, float | int | str]]:
    """
    Flatten nested summary statistics into row dictionaries.

    Parameters
    ----------
    summary : Mapping[str, Mapping[str, float | int]]
        Nested summary dictionary to flatten.

    Returns
    -------
    List[Dict[str, float | int | str]]
        Flattened summary rows ready for serialization.
    """
    rows: List[Dict[str, float | int | str]] = []
    for metric_name, stats_map in summary.items():
        row: Dict[str, float | int | str] = {"metric": metric_name}
        row.update(stats_map)
        rows.append(row)
    return rows
