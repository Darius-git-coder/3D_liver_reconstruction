from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from inpainting3d.utils import ensure_dir
from scripts.export_thesis_qualitative_figures import (
    configure_matplotlib,
    create_comparison_figure,
    create_large_matrix_figure,
    create_volume_render_figure,
    resolve_slice_descriptor,
    slugify,
)
from scripts.generate_qualitative_casebook import (
    build_dataset_sample,
    load_eval_artifacts,
    parse_models,
    pick_default_compare_label,
    pick_default_focus_label,
    resolve_model_map,
    select_cases,
    validate_metric,
)
from scripts.evaluate_ultrasound_distribution import load_model_spec
from scripts.train_inpainting import clamp_prediction
from scripts.visualize_model_comparison import resolve_eval_files, to_metrics


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_prediction_cached(
    x: torch.Tensor,
    *,
    model: torch.nn.Module,
    hard_constraint: bool,
    device: torch.device,
) -> Any:
    x_batched = x.unsqueeze(0).to(device)
    sparse = x_batched[:, 0:1, ...]
    mask = x_batched[:, 1:2, ...]
    with torch.no_grad():
        pred = clamp_prediction(model(x_batched), sparse=sparse, mask=mask, hard_constraint=hard_constraint)
    return pred[0, 0].detach().cpu().numpy().astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference_eval_dir", type=str, required=True)
    ap.add_argument("--slices", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--select_label", type=str, default="")
    ap.add_argument("--compare_label", type=str, default="")
    ap.add_argument("--metric", type=str, default="rmse_missing")
    ap.add_argument("--selector", type=str, default="mean", choices=["mean", "median"])
    ap.add_argument("--surface_threshold", type=float, default=0.10)
    ap.add_argument("--no_hard_constraint", dest="hard_constraint", action="store_false")
    ap.set_defaults(hard_constraint=True)
    args = ap.parse_args()

    configure_matplotlib()

    reference_eval_dir = os.path.normpath(os.path.abspath(args.reference_eval_dir))
    frame, repro, summary = load_eval_artifacts(reference_eval_dir)
    validate_metric(frame, args.metric)
    models = parse_models(repro, summary)
    model_map = resolve_model_map(models)

    focus_label = args.select_label.strip() or pick_default_focus_label(models)
    compare_label = args.compare_label.strip() or pick_default_compare_label(models, focus_label)
    if focus_label not in model_map:
        raise RuntimeError(f"Unknown focus label {focus_label!r}.")
    if not compare_label or compare_label not in model_map:
        raise RuntimeError("A valid comparison label is required.")

    args_payload = repro.get("args")
    if not isinstance(args_payload, dict):
        raise RuntimeError("Reference reproducibility.json does not contain an 'args' object.")

    selected = select_cases(
        frame,
        focus_label=focus_label,
        compare_label=compare_label,
        metric=str(args.metric),
        selector=str(args.selector),
        lower_is_better=True,
    )

    files = resolve_eval_files(
        data_glob=str(args_payload.get("data", "")),
        split_file=str(args_payload.get("split_file", "")),
        split=str(args_payload.get("split", "test")),
    )
    run_name = slugify(os.path.basename(os.path.dirname(reference_eval_dir)) if os.path.basename(reference_eval_dir).lower() == "eval" else os.path.basename(reference_eval_dir))
    out_dir = os.path.normpath(os.path.abspath(args.out)) if args.out else os.path.join(reference_eval_dir, "thesis_qualitative_multislice")
    ensure_dir(out_dir)

    focus_model = model_map[focus_label]
    compare_model = model_map[compare_label]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focus_model_loaded = load_model_spec(focus_model, init_feat=int(args_payload.get("init_feat", 32)), device=device)
    compare_model_loaded = load_model_spec(compare_model, init_feat=int(args_payload.get("init_feat", 32)), device=device)

    outputs: List[Dict[str, Any]] = []
    for slice_count in args.slices:
        slice_count = int(slice_count)
        override_args = dict(args_payload)
        override_args["slices"] = slice_count
        override_args["slices_min"] = slice_count
        override_args["slices_max"] = slice_count
        _, slice_label, slice_slug = resolve_slice_descriptor(override_args)

        slice_out_dir = os.path.join(out_dir, slice_slug)
        ensure_dir(slice_out_dir)

        for item in selected:
            x, y, meta = build_dataset_sample(
                files=files,
                case_id=item.case_id,
                seed=item.eval_seed,
                args_payload=override_args,
            )
            gt = y[0].numpy().astype("float32")
            sparse = x[0].numpy().astype("float32")
            mask = x[1].numpy().astype("float32")

            pred_focus = run_prediction_cached(
                x,
                model=focus_model_loaded,
                hard_constraint=bool(args.hard_constraint),
                device=device,
            )
            pred_compare = run_prediction_cached(
                x,
                model=compare_model_loaded,
                hard_constraint=bool(args.hard_constraint),
                device=device,
            )

            focus_metrics = to_metrics(pred_focus, gt, mask)
            compare_metrics = to_metrics(pred_compare, gt, mask)

            prefix = "__".join(
                [
                    run_name,
                    slice_slug,
                    slugify(item.archetype),
                    slugify(item.case_id),
                    f"seed_{item.eval_seed}",
                    slugify(compare_label),
                    "vs",
                    slugify(focus_label),
                ]
            )
            matrix_path = os.path.join(slice_out_dir, prefix + "__grosse_matrix.pdf")
            compare_path = os.path.join(slice_out_dir, prefix + "__vergleich.pdf")
            render_path = os.path.join(slice_out_dir, prefix + "__volumenrendering.pdf")

            create_large_matrix_figure(
                out_path=matrix_path,
                case_id=item.case_id,
                archetype=item.archetype,
                slice_label=slice_label,
                eval_seed=item.eval_seed,
                gt=gt,
                sparse=sparse,
                mask=mask,
                pred_base=pred_compare,
                pred_focus=pred_focus,
                base_mean=compare_metrics,
                focus_mean=focus_metrics,
            )
            create_comparison_figure(
                out_path=compare_path,
                case_id=item.case_id,
                archetype=item.archetype,
                slice_label=slice_label,
                eval_seed=item.eval_seed,
                gt=gt,
                mask=mask,
                pred_base=pred_compare,
                pred_focus=pred_focus,
                base_mean=compare_metrics,
                focus_mean=focus_metrics,
            )
            create_volume_render_figure(
                out_path=render_path,
                case_id=item.case_id,
                archetype=item.archetype,
                slice_label=slice_label,
                eval_seed=item.eval_seed,
                sparse=sparse,
                gt=gt,
                pred_base=pred_compare,
                pred_focus=pred_focus,
                base_mean=compare_metrics,
                focus_mean=focus_metrics,
                threshold=float(args.surface_threshold),
            )

            outputs.append(
                {
                    "slice_count": slice_count,
                    "slice_label": slice_label,
                    "archetype": item.archetype,
                    "case_id": item.case_id,
                    "eval_seed": int(item.eval_seed),
                    "matrix_pdf": os.path.normpath(matrix_path),
                    "comparison_pdf": os.path.normpath(compare_path),
                    "volume_pdf": os.path.normpath(render_path),
                    "focus_metrics": focus_metrics,
                    "compare_metrics": compare_metrics,
                    "meta": {
                        "case_id": meta.get("case_id"),
                        "path": meta.get("path"),
                        "slice_geometry": meta.get("slice_geometry"),
                        "dataset_index": meta.get("dataset_index"),
                    },
                }
            )

    manifest = {
        "reference_eval_dir": reference_eval_dir,
        "focus_label": focus_label,
        "compare_label": compare_label,
        "metric": str(args.metric),
        "selector": str(args.selector),
        "selected_cases_from_reference_eval": [
            {
                "archetype": item.archetype,
                "case_id": item.case_id,
                "eval_seed": int(item.eval_seed),
                "path": item.path,
            }
            for item in selected
        ],
        "outputs": outputs,
    }
    save_json(os.path.join(out_dir, "thesis_multislice_manifest.json"), manifest)
    print(json.dumps({"out": out_dir, "num_outputs": len(outputs)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
