from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inpainting3d.repro import build_repro_metadata, get_git_info
from inpainting3d.splits import create_split_manifest, resolve_case_paths, write_split_manifest


def save_json(path: str, payload: dict) -> None:
    """
    Write a JSON document to disk.
    
    Parameters
    ----------
    path : str
        Filesystem path to the input artifact.
    payload : dict
        Structured data that will be serialized as JSON.
    
    Returns
    -------
    None
        The JSON artifact is written to disk.
    """
    with open(path, "w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Execute the command-line entry point for this script.
    
    Returns
    -------
    None
        This function is executed for its side effects.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="./runs/data_split.json")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    args = ap.parse_args()

    files = resolve_case_paths(args.data)
    if not files:
        raise RuntimeError("Keine Dateien gefunden.")

    manifest = create_split_manifest(
        files,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        source_glob=args.data,
        project_root=ROOT,
        git_info=get_git_info(ROOT),
    )
    write_split_manifest(manifest, args.out)
    save_json(
        os.path.join(os.path.dirname(args.out) or ".", "split_reproducibility.json"),
        build_repro_metadata(vars(args), split_manifest_path=args.out, cwd=ROOT),
    )

    print(f"Split gespeichert: {os.path.normpath(os.path.abspath(args.out))}")
    print("Anzahlen:")
    for split_name, count in manifest["counts"].items():
        print(f"  {split_name:8s}: {count}")


if __name__ == "__main__":
    main()
