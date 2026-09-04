from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inpainting3d.stats import summarize_metric_values


DEFAULT_METRICS = [
    "rmse_missing",
    "mae_missing",
    "ssim_missing",
    "psnr_missing",
    "rmse_all",
    "mae_all",
    "ssim_all",
]

METADATA_COLUMNS = {
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_rows_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                fieldnames.append(key_str)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_input_spec(spec: str) -> Dict[str, str]:
    if "=" not in spec:
        raise ValueError(f"Input spec must look like method[:seed]=eval_dir, got {spec!r}.")
    left, eval_dir = spec.split("=", 1)
    if ":" in left:
        method, seed = left.split(":", 1)
    else:
        method, seed = left, "single"
    method = method.strip()
    seed = str(seed).strip() or "single"
    eval_dir = eval_dir.strip()
    if not method:
        raise ValueError(f"Missing method name in input spec {spec!r}.")
    if not eval_dir:
        raise ValueError(f"Missing eval directory in input spec {spec!r}.")
    return {"method": method, "train_seed": seed, "eval_dir": eval_dir}


def load_manifest_inputs(path: str) -> List[Dict[str, str]]:
    payload = json.load(open(path, "r", encoding="utf-8"))
    entries = payload.get("evaluations", [])
    inputs: List[Dict[str, str]] = []
    for entry in entries:
        eval_dir = entry.get("eval_dir") or entry.get("out")
        method = entry.get("method")
        if not eval_dir or not method:
            continue
        inputs.append(
            {
                "method": str(method),
                "train_seed": str(entry.get("seed", "single")),
                "eval_dir": str(eval_dir),
            }
        )
    return inputs


def collect_inputs(args: argparse.Namespace) -> List[Dict[str, str]]:
    inputs: List[Dict[str, str]] = []
    for spec in args.input or []:
        inputs.append(parse_input_spec(spec))
    for manifest_path in args.manifest or []:
        inputs.extend(load_manifest_inputs(manifest_path))
    if not inputs:
        raise RuntimeError("Provide at least two --input method[:seed]=eval_dir entries or a --manifest.")
    return inputs


def load_case_metrics(inputs: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in inputs:
        eval_dir = os.path.normpath(str(item["eval_dir"]))
        metrics_path = os.path.join(eval_dir, "metrics_per_case.csv")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"metrics_per_case.csv not found in {eval_dir!r}.")
        for row in read_csv_rows(metrics_path):
            row["method"] = str(item["method"])
            row["train_seed"] = str(item.get("train_seed", "single"))
            row["source_eval_dir"] = eval_dir
            rows.append(row)
    if not rows:
        raise RuntimeError("No case-level metric rows were loaded.")
    return rows


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def infer_metric_columns(rows: Sequence[Mapping[str, Any]], requested: Sequence[str] | None) -> List[str]:
    columns: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                columns.append(key_str)

    numeric_columns = [
        key
        for key in columns
        if key not in METADATA_COLUMNS and any(to_float(row.get(key)) is not None for row in rows)
    ]
    if requested:
        missing = [metric for metric in requested if metric not in numeric_columns]
        if missing:
            known = ", ".join(numeric_columns)
            raise RuntimeError(f"Requested metrics not found: {missing}. Available numeric metrics: {known}")
        return list(requested)

    ordered = [metric for metric in DEFAULT_METRICS if metric in numeric_columns]
    ordered.extend(metric for metric in numeric_columns if metric not in ordered)
    return ordered


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    repeats: int,
    confidence: float,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot bootstrap an empty value sequence.")
    if arr.size == 1 or repeats <= 0:
        mean = float(np.mean(arr))
        return mean, mean

    boot_means = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boot_means[index] = np.mean(sample)
    alpha = 100.0 * (1.0 - confidence) / 2.0
    return (
        float(np.percentile(boot_means, alpha)),
        float(np.percentile(boot_means, 100.0 - alpha)),
    )


def grouped(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, ""))].append(row)
    return dict(out)


def summarize_methods(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    confidence: float,
    bootstrap_repeats: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    summary_rows: List[Dict[str, Any]] = []
    for method, method_rows in sorted(grouped(rows, "method").items()):
        for metric in metrics:
            values = [to_float(row.get(metric)) for row in method_rows]
            values = [value for value in values if value is not None]
            if not values:
                continue
            metric_summary = summarize_metric_values(values, confidence=confidence)
            boot_low, boot_high = bootstrap_mean_ci(
                values,
                rng=rng,
                repeats=bootstrap_repeats,
                confidence=confidence,
            )
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    **metric_summary,
                    "bootstrap_ci_low": boot_low,
                    "bootstrap_ci_high": boot_high,
                }
            )
    return summary_rows


def metric_direction(metric: str) -> str:
    lower_prefixes = ("mae_", "rmse_", "mse_", "loss", "error")
    higher_prefixes = ("psnr_", "ssim_", "dice", "iou", "accuracy")
    if metric.startswith(lower_prefixes) or "error" in metric:
        return "lower_is_better"
    if metric.startswith(higher_prefixes):
        return "higher_is_better"
    return "lower_is_better"


def case_identifier(row: Mapping[str, Any]) -> str:
    for key in ("case_id", "path", "index"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise RuntimeError(f"Cannot determine a case identifier for row: {row}")


def pair_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return str(row.get("train_seed", "single")), case_identifier(row)


def values_by_key(rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[Tuple[str, str], float]:
    values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        value = to_float(row.get(metric))
        if value is None:
            continue
        values[pair_key(row)].append(value)
    return {key: float(np.mean(group_values)) for key, group_values in values.items()}


def wilcoxon_p_value(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2 or np.allclose(arr, 0.0):
        return 1.0
    try:
        return float(scipy_stats.wilcoxon(arr, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def paired_comparisons(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    baseline: str,
    confidence: float,
    bootstrap_repeats: int,
    seed: int,
) -> List[Dict[str, Any]]:
    by_method = grouped(rows, "method")
    if baseline not in by_method:
        raise RuntimeError(f"Baseline method {baseline!r} is missing. Available: {sorted(by_method)}")

    rng = np.random.default_rng(seed + 1)
    pairwise_rows: List[Dict[str, Any]] = []
    for method in sorted(name for name in by_method if name != baseline):
        for metric in metrics:
            base_values = values_by_key(by_method[baseline], metric)
            comp_values = values_by_key(by_method[method], metric)
            shared_keys = sorted(set(base_values) & set(comp_values))
            if not shared_keys:
                continue

            direction = metric_direction(metric)
            baseline_arr = np.asarray([base_values[key] for key in shared_keys], dtype=np.float64)
            comparator_arr = np.asarray([comp_values[key] for key in shared_keys], dtype=np.float64)
            raw_delta = comparator_arr - baseline_arr
            if direction == "lower_is_better":
                improvement = baseline_arr - comparator_arr
            else:
                improvement = comparator_arr - baseline_arr

            boot_low, boot_high = bootstrap_mean_ci(
                improvement.tolist(),
                rng=rng,
                repeats=bootstrap_repeats,
                confidence=confidence,
            )
            std = float(np.std(improvement, ddof=1)) if improvement.size > 1 else 0.0
            pairwise_rows.append(
                {
                    "baseline": baseline,
                    "comparator": method,
                    "metric": metric,
                    "direction": direction,
                    "n_pairs": int(improvement.size),
                    "baseline_mean": float(np.mean(baseline_arr)),
                    "comparator_mean": float(np.mean(comparator_arr)),
                    "raw_delta_mean_comparator_minus_baseline": float(np.mean(raw_delta)),
                    "mean_improvement_positive_is_better": float(np.mean(improvement)),
                    "median_improvement_positive_is_better": float(np.median(improvement)),
                    "bootstrap_ci_low": boot_low,
                    "bootstrap_ci_high": boot_high,
                    "wilcoxon_p_two_sided": wilcoxon_p_value(improvement.tolist()),
                    "win_rate": float(np.mean(improvement > 0.0)),
                    "standardized_mean_improvement": float(np.mean(improvement) / std) if std > 0.0 else 0.0,
                }
            )
    return pairwise_rows


def fmt(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return str(value)
    return f"{number:.6g}"


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_markdown_report(
    path: str,
    *,
    metrics: Sequence[str],
    baseline: str,
    method_summary: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, str]],
) -> None:
    selected_summary = [
        row
        for row in method_summary
        if row["metric"] in metrics[: min(len(metrics), len(DEFAULT_METRICS))]
    ]
    selected_pairwise = [
        row
        for row in pairwise_rows
        if row["metric"] in metrics[: min(len(metrics), len(DEFAULT_METRICS))]
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Scientific Method Comparison\n\n")
        handle.write(
            "Positive pairwise improvement means that the comparator is better than the baseline. "
            "For MAE/RMSE/error metrics lower is better; for SSIM/PSNR/Dice/IoU higher is better.\n\n"
        )
        handle.write(f"- Baseline: `{baseline}`\n")
        handle.write(f"- Metrics: {', '.join(f'`{metric}`' for metric in metrics)}\n")
        handle.write(f"- Inputs: {len(inputs)} evaluation folders\n\n")
        handle.write("## Inputs\n\n")
        handle.write(markdown_table(inputs, ["method", "train_seed", "eval_dir"]))
        handle.write("\n\n## Method Summary\n\n")
        handle.write(
            markdown_table(
                selected_summary,
                [
                    "method",
                    "metric",
                    "n",
                    "mean",
                    "median",
                    "std",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                ],
            )
        )
        handle.write("\n\n## Pairwise Tests\n\n")
        handle.write(
            markdown_table(
                selected_pairwise,
                [
                    "baseline",
                    "comparator",
                    "metric",
                    "n_pairs",
                    "mean_improvement_positive_is_better",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "wilcoxon_p_two_sided",
                    "win_rate",
                ],
            )
        )
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and statistically compare per-case evaluation metrics from multiple methods."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Evaluation folder as method[:train_seed]=path. Repeat for each method/seed.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Suite manifest JSON with an 'evaluations' list. Can be repeated.",
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="resunet")
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    inputs = collect_inputs(args)
    if len({item["method"] for item in inputs}) < 2:
        raise RuntimeError("At least two methods are required for a method comparison.")

    rows = load_case_metrics(inputs)
    metrics = infer_metric_columns(rows, args.metrics)
    method_summary = summarize_methods(
        rows,
        metrics,
        confidence=args.confidence,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    pairwise_rows = paired_comparisons(
        rows,
        metrics,
        baseline=args.baseline,
        confidence=args.confidence,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )

    write_rows_csv(os.path.join(args.out, "all_case_metrics.csv"), rows)
    write_rows_csv(os.path.join(args.out, "method_metric_summary.csv"), method_summary)
    write_rows_csv(os.path.join(args.out, "pairwise_tests.csv"), pairwise_rows)
    save_json(
        os.path.join(args.out, "scientific_comparison_summary.json"),
        {
            "settings": vars(args),
            "inputs": inputs,
            "metrics": metrics,
            "method_summary": method_summary,
            "pairwise_tests": pairwise_rows,
        },
    )
    write_markdown_report(
        os.path.join(args.out, "scientific_comparison_summary.md"),
        metrics=metrics,
        baseline=args.baseline,
        method_summary=method_summary,
        pairwise_rows=pairwise_rows,
        inputs=inputs,
    )

    print(f"Wrote scientific comparison to {os.path.normpath(args.out)}")
    print(f"Compared methods: {', '.join(sorted({item['method'] for item in inputs}))}")
    print(f"Metrics: {', '.join(metrics)}")


if __name__ == "__main__":
    main()
