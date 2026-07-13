"""File I/O helpers: infer flags from filenames, evaluate files/directories,
combine per-file metrics into a single JSON.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from multibbq.metrics.scorer import eval_visual_language, eval_visual_only

PathLike = Union[str, os.PathLike]


def infer_flags_from_filename(path: PathLike) -> Tuple[str, bool, bool]:
    """Parse mode / ambig / neg from a BBQ-style result filename.

    Filenames encode the setting via tokens joined by underscores, e.g.:
        MiniCPM_4_5_visual_language_negative_ambiguous.json
        llava_34b_visual_only_nonnegative_ambiguous.json

    Returns:
        (mode, ambig, neg) where mode ∈ {"visual_only", "visual_language"}.

    Raises:
        ValueError if the mode cannot be determined.
    """
    stem = Path(path).stem
    tokens = stem.split("_")
    # Explicit modality tokens win over 'text' so a model NAME containing a bare
    # 'text' token cannot misroute a visual_only file.
    # 'text' = text-only LLM eval; scored with the same (context + question)
    # logic as visual_language (BBQ-style ambiguous / disambiguous).
    if "visual_language" in stem:
        mode = "visual_language"
    elif "visual_only" in stem:
        mode = "visual_only"
    elif "text" in tokens:
        mode = "visual_language"
    else:
        raise ValueError(
            f"Cannot infer mode from filename {path!r}; "
            "expected 'visual_only', 'visual_language', or 'text' in the name."
        )
    # 'nonnegative' contains 'negative' as substring - check token membership, not substring.
    neg = "negative" in tokens
    ambig = "ambiguous" in tokens
    return mode, ambig, neg


def eval_file(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    tail_slice: Optional[int] = None,
    mode: Optional[str] = None,
    ambig: Optional[bool] = None,
    neg: Optional[bool] = None,
) -> dict:
    """Score a single results JSON.

    Reads `input_path` (must contain a top-level "data" list), computes metrics,
    and optionally writes an enriched copy to `output_path` (mirroring the
    notebook's `*_w_metrics.json` behavior).

    Any of `mode`, `ambig`, `neg` left as None will be inferred from the
    filename.

    Returns:
        The metrics dict (same schema as scorer functions).
    """
    input_path = Path(input_path)
    if mode is None or ambig is None or neg is None:
        inferred_mode, inferred_ambig, inferred_neg = infer_flags_from_filename(input_path)
        mode = inferred_mode if mode is None else mode
        ambig = inferred_ambig if ambig is None else ambig
        neg = inferred_neg if neg is None else neg

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload

    if mode == "visual_only":
        metrics = eval_visual_only(rows, neg=neg, tail_slice=tail_slice)
    elif mode == "visual_language":
        metrics = eval_visual_language(rows, ambig=ambig, neg=neg, tail_slice=tail_slice)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, dict):
            payload["metrics"] = metrics
            out_payload = payload
        else:
            out_payload = {"data": rows, "metrics": metrics}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f)

    return metrics


def eval_directory(
    input_dir: PathLike,
    output_dir: PathLike,
    tail_slice: Optional[int] = None,
    pattern_suffix: str = ".json",
    skip_existing: bool = False,
) -> Dict[str, dict]:
    """Score every result file under `input_dir` and mirror the tree to `output_dir`.

    For each `<name>.json` found, writes `<name>_w_metrics.json` under the
    same relative subpath in `output_dir`. Files whose names don't parse
    (missing visual_only/visual_language tag) are skipped with a warning.

    Args:
        input_dir: Root of the results tree.
        output_dir: Root of the output tree. Created if missing.
        tail_slice: See `parse_pred`.
        pattern_suffix: Only files ending with this suffix are considered.
        skip_existing: If True, do not re-score files whose `_w_metrics.json`
            already exists.

    Returns:
        {relative_path: metrics_dict, ...} for every file that was scored.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    scored: Dict[str, dict] = {}

    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if not filename.endswith(pattern_suffix):
                continue
            if filename.endswith("_w_metrics.json"):
                continue

            in_path = Path(dirpath) / filename
            rel = in_path.relative_to(input_dir)
            stem = in_path.stem
            out_path = output_dir / rel.parent / f"{stem}_w_metrics.json"

            if skip_existing and out_path.exists():
                continue

            try:
                metrics = eval_file(in_path, out_path, tail_slice=tail_slice)
            except ValueError as e:
                print(f"[skip] {in_path}: {e}")
                continue
            scored[str(rel)] = metrics

    return scored


def combine_metrics(root_directory: PathLike, output_path: PathLike) -> int:
    """Aggregate every `*_w_metrics.json` under a directory into one JSON array.

    Each entry is `{"json_name": <relative_path>, ...<metrics fields>}`,
    matching the shape consumed by `aggregate.py`.

    Returns:
        Number of files aggregated.
    """
    root_directory = Path(root_directory)
    output_path = Path(output_path)
    aggregated = []

    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if not filename.endswith("_w_metrics.json"):
                continue
            file_path = Path(dirpath) / filename
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"[skip] cannot parse JSON: {file_path}")
                continue

            if "metrics" not in data:
                print(f"[skip] no 'metrics' key: {file_path}")
                continue

            rel = file_path.relative_to(root_directory).as_posix()
            aggregated.append({"json_name": rel, **data["metrics"]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=4)
    return len(aggregated)
