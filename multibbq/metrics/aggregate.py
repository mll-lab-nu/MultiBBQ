"""Aggregate combined_metrics.json into CSV summaries + FS_total / BS_total.

Pipeline (run `run_pipeline` for the full flow, or call individual steps):

    combined_metrics.json
        │
        ├─ create_overall_metrics_csv         → overall_metrics_summary_sorted.csv
        ├─ create_all_category_metrics_csvs   → {age,gender,race,religion}_metrics_summary_sorted.csv
        ├─ create_combined_summary_csv        → model_category_average_scores.csv
        │
        └─ cal_total_scores (per sub-CSV)     → *_metrics_details.csv (adds FS_total/BS_total)
                │
                └─ combine_category_totals    → category_total_scores_summary.csv

The MultiIndex column layout (`Visual Language`/`Visual Only` × `Ambiguous/Disambiguous`
× `Negative/Nonnegative` × `Fairness/Bias`) is BBQ-shaped; if you use this
package with your own benchmark, either match that shape or write your own
aggregator on top of `combine_metrics` output.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

PathLike = Union[str, os.PathLike]


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", s)]


# ---------------------------------------------------------------------------
# 3-term harmonic mean over (FLAm, FLDis, FVAm)
# ---------------------------------------------------------------------------

def hmean_flexible(a: pd.Series, b: pd.Series, c: pd.Series, eps: float = 1e-10) -> pd.Series:
    """Harmonic mean that gracefully drops missing terms (3→H3, 2→H2, 1→identity)."""
    vals = pd.concat([a, b, c], axis=1)
    cnt = vals.notna().sum(axis=1)
    res = pd.Series(index=vals.index, dtype=float)

    m3 = cnt == 3
    res[m3] = 3 * a[m3] * b[m3] * c[m3] / (a[m3] * b[m3] + b[m3] * c[m3] + a[m3] * c[m3] + eps)

    m_ab = (cnt == 2) & a.notna() & b.notna()
    res[m_ab] = 2 * a[m_ab] * b[m_ab] / (a[m_ab] + b[m_ab] + eps)

    m_bc = (cnt == 2) & b.notna() & c.notna()
    res[m_bc] = 2 * b[m_bc] * c[m_bc] / (b[m_bc] + c[m_bc] + eps)

    m_ac = (cnt == 2) & a.notna() & c.notna()
    res[m_ac] = 2 * a[m_ac] * c[m_ac] / (a[m_ac] + c[m_ac] + eps)

    m1 = cnt == 1
    res[m1] = vals[m1].sum(axis=1, skipna=True)

    return res


def cal_total_scores(input_path: PathLike, output_path: PathLike, include_mid: bool = False) -> None:
    """Compute FS_total and BS_total on a per-model summary CSV.

    Reads a CSV with 3-level MultiIndex columns produced by
    `create_overall_metrics_csv` / `create_all_category_metrics_csvs`,
    averages Neg/Nonneg pairs to FLAm / FLDis / FVAm, and combines those
    via flexible harmonic mean into FS_total / BS_total.
    """
    df = pd.read_csv(input_path, header=[0, 1, 2], index_col=0)

    fl_amb_fs_avg = (
        df[("Visual Language", "Ambiguous Negative", "Fairness Score")]
        + df[("Visual Language", "Ambiguous Nonnegative", "Fairness Score")]
    ) / 2
    fl_dis_fs_avg = (
        df[("Visual Language", "Disambiguous Negative", "Fairness Score")]
        + df[("Visual Language", "Disambiguous Nonnegative", "Fairness Score")]
    ) / 2
    vo_amb_fs_avg = (
        df[("Visual Only", "Ambiguous Negative", "Fairness Score")]
        + df[("Visual Only", "Ambiguous Nonnegative", "Fairness Score")]
    ) / 2

    fl_amb_bs_avg = (
        df[("Visual Language", "Ambiguous Negative", "Bias Score")]
        + df[("Visual Language", "Ambiguous Nonnegative", "Bias Score")]
    ) / 2
    fl_dis_bs_avg = (
        df[("Visual Language", "Disambiguous Negative", "Bias Score")]
        + df[("Visual Language", "Disambiguous Nonnegative", "Bias Score")]
    ) / 2
    vo_amb_bs_avg = (
        df[("Visual Only", "Ambiguous Negative", "Bias Score")]
        + df[("Visual Only", "Ambiguous Nonnegative", "Bias Score")]
    ) / 2

    if include_mid:
        df[("FS_mid", "FLAm", "")] = fl_amb_fs_avg
        df[("FS_mid", "FLDis", "")] = fl_dis_fs_avg
        df[("FS_mid", "FVAm", "")] = vo_amb_fs_avg
        df[("BS_mid", "FLAm", "")] = fl_amb_bs_avg
        df[("BS_mid", "FLDis", "")] = fl_dis_bs_avg
        df[("BS_mid", "FVAm", "")] = vo_amb_bs_avg

    df[("FS_total", "", "")] = hmean_flexible(fl_amb_fs_avg, fl_dis_fs_avg, vo_amb_fs_avg)
    # Bias is "lower is better", so combine via 1 - hmean(1 - bs).
    df[("BS_total", "", "")] = 1 - hmean_flexible(1 - fl_amb_bs_avg, 1 - fl_dis_bs_avg, 1 - vo_amb_bs_avg)

    fs_vo_cols = [c for c in df.columns if "Fairness Score" in c[2] and "Visual Only" in c[0]]
    fs_vl_cols = [c for c in df.columns if "Fairness Score" in c[2] and "Visual Language" in c[0]]
    bs_vo_cols = [c for c in df.columns if "Bias Score" in c[2] and "Visual Only" in c[0]]
    bs_vl_cols = [c for c in df.columns if "Bias Score" in c[2] and "Visual Language" in c[0]]

    mid_fs_cols = [("FS_mid", "FLAm", ""), ("FS_mid", "FLDis", ""), ("FS_mid", "FVAm", "")]
    mid_bs_cols = [("BS_mid", "FLAm", ""), ("BS_mid", "FLDis", ""), ("BS_mid", "FVAm", "")]

    final_cols_order = (
        fs_vo_cols
        + fs_vl_cols
        + [("FS_total", "", "")]
        + bs_vo_cols
        + bs_vl_cols
        + [("BS_total", "", "")]
        + (mid_fs_cols if include_mid else [])
        + (mid_bs_cols if include_mid else [])
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[final_cols_order].to_csv(output_path)


# ---------------------------------------------------------------------------
# combined_metrics.json → CSVs
# ---------------------------------------------------------------------------

# Greedy prefix: the LAST modality token wins, so a model NAME containing a bare
# 'text' token cannot misroute a visual_only / visual_language file.
_NAME_PATTERN = re.compile(r"^(.*)_(visual_language|visual_only|text)_(.*)_w_metrics\.json$")


def parse_json_name(json_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse a combined_metrics entry's `json_name` into (model, category, sub_category).

    Example: `openbmb/MiniCPM-V-4_5/MiniCPM_4_5_visual_language_negative_ambiguous_w_metrics.json`
        → ("openbmb/MiniCPM-V-4_5", "Visual Language", "Ambiguous Negative").
    """
    match = _NAME_PATTERN.match(json_name)
    if not match:
        return None, None, None

    model_name_raw, category_raw, scenario_part = match.groups()
    # text-only LLM eval has no visual modality; it occupies the language
    # scenario slot, so FS_Total/BS_Total reduce to the 2-scenario (Am + Dis)
    # harmonic mean (the Visual Only columns stay empty -> hmean_flexible uses 2 terms).
    if category_raw == "text":
        category = "Visual Language"
    else:
        category = category_raw.replace("_", " ").title()
    model_name = os.path.dirname(model_name_raw)

    if "ambiguous" in scenario_part and "disambiguous" not in scenario_part:
        prefix = "Ambiguous"
    elif "disambiguous" in scenario_part:
        prefix = "Disambiguous"
    else:
        prefix = "Ambiguous"

    if "negative" in scenario_part and "nonnegative" not in scenario_part:
        suffix = "Negative"
    else:
        suffix = "Nonnegative"

    return model_name, category, f"{prefix} {suffix}"


def _load_json(path: PathLike) -> Optional[List[Dict[str, Any]]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[aggregate] file not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[aggregate] JSON parse error in {path}: {e}")
        return None


def _default_column_index() -> pd.MultiIndex:
    metrics_order = ["Fairness Score", "Bias Score"]
    vl_subs = [
        "Ambiguous Negative", "Ambiguous Nonnegative",
        "Disambiguous Negative", "Disambiguous Nonnegative",
    ]
    vo_subs = ["Ambiguous Negative", "Ambiguous Nonnegative"]
    tuples: List[Tuple[str, str, str]] = []
    for sub in vl_subs:
        for m in metrics_order:
            tuples.append(("Visual Language", sub, m))
    for sub in vo_subs:
        for m in metrics_order:
            tuples.append(("Visual Only", sub, m))
    return pd.MultiIndex.from_tuples(tuples, names=["Category", "SubCategory", "Metric"])


def create_overall_metrics_csv(json_file_path: PathLike, output_csv_path: PathLike) -> None:
    """Extract per-model `overall` scores into a MultiIndex CSV."""
    data = _load_json(json_file_path)
    if data is None:
        return

    processed: Dict[str, Dict[Tuple[str, str, str], float]] = defaultdict(dict)
    for entry in data:
        json_name = entry.get("json_name")
        container = entry.get("metrics", entry)
        overall = container.get("overall", {})
        fs = overall.get("fairness_score")
        bs = overall.get("bias_score")
        if not (json_name and fs is not None and bs is not None):
            continue
        model, category, sub_category = parse_json_name(json_name)
        if not model:
            print(f"[aggregate] cannot parse json_name: {json_name}")
            continue
        processed[model][(category, sub_category, "Fairness Score")] = fs
        processed[model][(category, sub_category, "Bias Score")] = bs

    if not processed:
        print("[aggregate] no valid overall data")
        return

    df = pd.DataFrame.from_dict(processed, orient="index")
    df = df.reindex(sorted(df.index, key=natural_sort_key))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.reindex(columns=_default_column_index())
    df = df.reset_index().rename(columns={"index": "Model"})

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")


def create_all_category_metrics_csvs(json_file_path: PathLike, output_dir: PathLike) -> None:
    """One MultiIndex CSV per benchmark category (age / gender / race / religion / ...)."""
    data = _load_json(json_file_path)
    if data is None:
        return

    per_cat: Dict[str, Dict[str, Dict[Tuple[str, str, str], float]]] = defaultdict(lambda: defaultdict(dict))
    for entry in data:
        json_name = entry.get("json_name")
        by_category = entry.get("by_category", {})
        if not json_name or not by_category:
            continue
        model, main_category, sub_category = parse_json_name(json_name)
        if not model:
            print(f"[aggregate] cannot parse json_name: {json_name}")
            continue
        for cat_name, scores in by_category.items():
            fs = scores.get("fairness_score")
            bs = scores.get("bias_score")
            if fs is not None and bs is not None:
                per_cat[cat_name][model][(main_category, sub_category, "Fairness Score")] = fs
                per_cat[cat_name][model][(main_category, sub_category, "Bias Score")] = bs

    if not per_cat:
        print("[aggregate] no valid by_category data")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    col_index = _default_column_index()

    for cat_name, processed in per_cat.items():
        df = pd.DataFrame.from_dict(processed, orient="index")
        df = df.reindex(sorted(df.index, key=natural_sort_key))
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.reindex(columns=col_index)
        df = df.reset_index().rename(columns={"index": "Model"})
        df.to_csv(output_dir / f"{cat_name}_metrics_summary_sorted.csv", index=False, encoding="utf-8-sig")


def create_combined_summary_csv(
    json_file_path: PathLike,
    output_csv_path: PathLike,
    subcategory_order: Optional[List[str]] = None,
) -> None:
    """Per-model average FS↑ / BS↓ per main category, under a 3-level column header.

    Args:
        json_file_path: combined_metrics.json.
        output_csv_path: output CSV.
        subcategory_order: title-cased benchmark subcategories to include, in
            column order. Any subcategory absent from this list is dropped
            from the output. Default: `["Gender", "Race", "Religion", "Age"]`
            (BBQ layout).
    """
    if subcategory_order is None:
        subcategory_order = ["Gender", "Race", "Religion", "Age"]
    data = _load_json(json_file_path)
    if data is None:
        return

    aggregator = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for entry in data:
        json_name = entry.get("json_name")
        by_category = entry.get("by_category")
        if not json_name or not by_category:
            continue
        model, main_category, _ = parse_json_name(json_name)
        if not model:
            continue
        for sub_category, scores in by_category.items():
            sub_title = sub_category.title()
            if scores.get("fairness_score") is not None:
                aggregator[main_category][model][sub_title]["Fairness Score"].append(scores["fairness_score"])
            if scores.get("bias_score") is not None:
                aggregator[main_category][model][sub_title]["Bias Score"].append(scores["bias_score"])

    if not aggregator:
        print("[aggregate] no valid by_category data")
        return

    final: Dict[str, Dict[Tuple[str, str, str], float]] = defaultdict(dict)
    for main_category, model_data in aggregator.items():
        for model, sub_categories in model_data.items():
            for sub_cat, metrics in sub_categories.items():
                fs_scores = metrics.get("Fairness Score", [])
                bs_scores = metrics.get("Bias Score", [])
                if fs_scores:
                    final[model][(main_category, "Fairness Score ↑", f"FS_{sub_cat}")] = sum(fs_scores) / len(fs_scores)
                if bs_scores:
                    final[model][(main_category, "Bias Score ↓", f"BS_{sub_cat}")] = sum(bs_scores) / len(bs_scores)

    if not final:
        print("[aggregate] no aggregated data")
        return

    df = pd.DataFrame.from_dict(final, orient="index")
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    level0 = ["Visual Language", "Visual Only"]
    level1 = ["Fairness Score ↑", "Bias Score ↓"]

    ordered = []
    for l0 in level0:
        for l1 in level1:
            prefix = "FS" if "Fairness" in l1 else "BS"
            for l2 in subcategory_order:
                col = (l0, l1, f"{prefix}_{l2}")
                if col in df.columns:
                    ordered.append(col)
    df = df.reindex(columns=ordered)
    df = df.reindex(sorted(df.index, key=natural_sort_key))
    df = df.reset_index().rename(columns={"index": "Model"})

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")


def combine_category_totals(
    metrics_dir: PathLike,
    categories: Optional[List[str]] = None,
    output_csv_path: Optional[PathLike] = None,
) -> None:
    """Merge FS_total / BS_total from each `*_metrics_details.csv` into one summary."""
    if categories is None:
        categories = ["overall", "age", "gender", "race", "religion"]
    metrics_dir = Path(metrics_dir)
    if output_csv_path is None:
        output_csv_path = metrics_dir / "category_total_scores_summary.csv"
    output_csv_path = Path(output_csv_path)

    def _read_one(cat: str) -> Optional[pd.DataFrame]:
        in_path = metrics_dir / f"{cat}_metrics_details.csv"
        if not in_path.is_file():
            print(f"[combine] skip missing: {in_path}")
            return None
        df = pd.read_csv(in_path, header=[0, 1, 2], index_col=0)

        def _resolve(prefix: str) -> Optional[Tuple[str, str, str]]:
            key = (prefix, "", "")
            if key in df.columns:
                return key
            cands = [c for c in df.columns if isinstance(c, tuple) and c[0] == prefix]
            return cands[0] if cands else None

        fs_key = _resolve("FS_total")
        bs_key = _resolve("BS_total")
        if fs_key is None or bs_key is None:
            print(f"[combine] missing FS_total / BS_total in {in_path}")
            return None

        fs = df[fs_key].copy()
        bs = df[bs_key].copy()
        fs.name = ("FS_total", cat)
        bs.name = ("BS_total", cat)
        out = pd.concat([fs, bs], axis=1)
        out.columns = pd.MultiIndex.from_tuples(out.columns, names=["Metric", "Category"])
        return out

    merged: Optional[pd.DataFrame] = None
    for cat in categories:
        part = _read_one(cat)
        if part is None:
            continue
        merged = part if merged is None else merged.join(part, how="outer")

    if merged is None or merged.empty:
        print("[combine] nothing to combine")
        return

    merged.index = merged.index.astype(str)
    merged = merged.reindex(sorted(merged.index, key=natural_sort_key))
    ordered = [(m, c) for m in ("FS_total", "BS_total") for c in categories if (m, c) in merged.columns]
    merged = merged[ordered]
    merged.index.name = "Model"
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv_path, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    combined_metrics_json: PathLike,
    csv_dir: PathLike,
    metrics_dir: PathLike,
    categories: Optional[List[str]] = None,
    include_mid: bool = False,
    subcategory_order: Optional[List[str]] = None,
) -> None:
    """Run every step: JSON → per-category CSVs → *_metrics_details → summary."""
    csv_dir = Path(csv_dir)
    metrics_dir = Path(metrics_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = ["gender", "race", "religion", "age"]

    create_overall_metrics_csv(combined_metrics_json, csv_dir / "overall_metrics_summary_sorted.csv")
    create_all_category_metrics_csvs(combined_metrics_json, csv_dir)
    create_combined_summary_csv(
        combined_metrics_json,
        csv_dir / "model_category_average_scores.csv",
        subcategory_order=subcategory_order,
    )

    for stem in ["overall", *categories]:
        in_path = csv_dir / f"{stem}_metrics_summary_sorted.csv"
        out_path = metrics_dir / f"{stem}_metrics_details.csv"
        if not in_path.is_file():
            print(f"[pipeline] missing input: {in_path}")
            continue
        cal_total_scores(in_path, out_path, include_mid=include_mid)

    combine_category_totals(
        metrics_dir=metrics_dir,
        categories=categories,
        output_csv_path=metrics_dir / "category_total_scores_summary.csv",
    )
