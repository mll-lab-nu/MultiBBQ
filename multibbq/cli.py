"""MultiBBQ command-line interface.

Subcommands:
    run          - model inference (needs the conda GPU environment)
    score        - score a results file or directory (fairness / bias / unk)
    combine      - walk *_w_metrics.json and produce combined_metrics.json
    aggregate    - combined_metrics.json → CSV summaries + FS_total / BS_total
    pipeline     - score → combine → aggregate in one shot

`run` imports the model stack (torch etc.) lazily, so the metric subcommands
work in a lightweight environment with only pandas installed.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

_SCORE_KEYS = ("fairness_score", "bias_score", "unk_rate")


def _filter_scores(metrics: dict, keep: Iterable[str]) -> dict:
    keep_set = set(keep)
    if keep_set == set(_SCORE_KEYS):
        return metrics
    def _prune(d: dict) -> dict:
        return {k: v for k, v in d.items() if k in keep_set}
    filtered = {"overall": _prune(metrics.get("overall", {}))}
    if "by_category" in metrics:
        filtered["by_category"] = {c: _prune(v) for c, v in metrics["by_category"].items()}
    return filtered


def _resolve_score_flag(flag: str) -> Iterable[str]:
    if flag == "all":
        return _SCORE_KEYS
    mapping = {"fairness": "fairness_score", "bias": "bias_score", "unk": "unk_rate"}
    return [mapping[flag]]


def cmd_run(args: argparse.Namespace) -> int:
    from multibbq import inference  # lazy: pulls in the model stack
    return inference.run(args)


def cmd_download(args: argparse.Namespace) -> int:
    from multibbq.hf import download
    return download(repo_id=args.repo, root=args.root,
                    primary=not args.no_primary,
                    realworld=args.realworld or args.all,
                    perturbations=args.perturbations or args.all)


def cmd_score(args: argparse.Namespace) -> int:
    from multibbq.metrics.io import eval_directory, eval_file

    input_path = Path(args.input)
    keep = _resolve_score_flag(args.score)

    if input_path.is_file():
        metrics = eval_file(input_path, args.output, tail_slice=args.tail_slice)
        filtered = _filter_scores(metrics, keep)
        json.dump(filtered, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if input_path.is_dir():
        if not args.output:
            print("--output <dir> is required when --input is a directory", file=sys.stderr)
            return 2
        scored = eval_directory(
            input_path,
            args.output,
            tail_slice=args.tail_slice,
            skip_existing=args.skip_existing,
        )
        print(f"scored {len(scored)} files → {args.output}")
        return 0

    print(f"input path does not exist: {input_path}", file=sys.stderr)
    return 2


def cmd_combine(args: argparse.Namespace) -> int:
    from multibbq.metrics.io import combine_metrics

    n = combine_metrics(args.input, args.output)
    print(f"combined {n} files → {args.output}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    from multibbq.metrics.aggregate import (
        cal_total_scores,
        combine_category_totals,
        create_all_category_metrics_csvs,
        create_combined_summary_csv,
        create_overall_metrics_csv,
    )

    csv_dir = Path(args.csv_dir)
    metrics_dir = Path(args.metrics_dir)
    create_overall_metrics_csv(args.input, csv_dir / "overall_metrics_summary_sorted.csv")
    create_all_category_metrics_csvs(args.input, csv_dir)
    create_combined_summary_csv(
        args.input,
        csv_dir / "model_category_average_scores.csv",
        subcategory_order=args.subcategory_order,
    )
    for stem in ["overall", *args.categories]:
        in_path = csv_dir / f"{stem}_metrics_summary_sorted.csv"
        out_path = metrics_dir / f"{stem}_metrics_details.csv"
        if not in_path.is_file():
            print(f"[skip] missing: {in_path}")
            continue
        cal_total_scores(in_path, out_path, include_mid=args.include_mid)
    combine_category_totals(metrics_dir=metrics_dir, categories=args.categories)
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    from multibbq.metrics.aggregate import run_pipeline
    from multibbq.metrics.io import combine_metrics, eval_directory

    tmp_combined = Path(args.output) / "combined_metrics.json"
    scored = eval_directory(args.input, Path(args.output) / "analysis", tail_slice=args.tail_slice)
    print(f"scored {len(scored)} files")
    n = combine_metrics(Path(args.output) / "analysis", tmp_combined)
    print(f"combined {n} files → {tmp_combined}")
    run_pipeline(
        tmp_combined,
        csv_dir=Path(args.output) / "csv_files",
        metrics_dir=Path(args.output) / "metrics_details",
        categories=args.categories,
        include_mid=args.include_mid,
        subcategory_order=args.subcategory_order,
    )
    return 0


def _add_metric_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--categories", nargs="+", default=["gender", "race", "religion", "age"])
    p.add_argument(
        "--subcategory-order", nargs="+", default=None,
        help="title-cased subcategories for model_category_average_scores.csv column order "
             "(default: Gender Race Religion Age)",
    )
    p.add_argument("--include-mid", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    from multibbq import __version__

    p = argparse.ArgumentParser(prog="multibbq", description=__doc__)
    p.add_argument("--version", action="version", version=f"multibbq {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    from multibbq.experiments import EXPERIMENTS, SYSTEM_MSGS  # light import (pure data)
    pr = sub.add_parser("run", help="run model inference for one evaluation setting")
    pr.add_argument("model_id", type=str, help="HuggingFace / API model id")
    pr.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="main",
                    help="which evaluation setting to run")
    pr.add_argument("--data_id", default="gpt_image_gen", help="gpt_image_gen or imagen4ultra_image_gen")
    pr.add_argument("--textual_context", default="true",
                    help="true=visual-language, false=visual-only")
    pr.add_argument("--ambiguous", default="true",
                    help="true=ambiguous, false=disambiguous context")
    pr.add_argument("--negative", default="true",
                    help="true=negative, false=non-negative questions")
    pr.add_argument("--img_aug_type", default="noise",
                    help="aug_img/img_label: noise|brightness|compression|contrast|resize_l|resize_s")
    pr.add_argument("--temperature", type=float, default=0.2,
                    help="temp experiment: 0.2|0.4|0.6|0.8|1.0")
    pr.add_argument("--reasoning_mode", default="reasoning", choices=list(SYSTEM_MSGS),
                    help="reasoning experiment: which system instruction to apply")
    pr.set_defaults(func=cmd_run)

    # download
    pd = sub.add_parser("download", help="download the released images and lay out ./data/images/")
    pd.add_argument("--repo", default="MLL-Lab/MultiBBQ", help="HuggingFace dataset id")
    pd.add_argument("--root", default=".", help="place the data/images/ tree under this directory")
    pd.add_argument("--realworld", action="store_true",
                    help="also fetch real_world_image/ (~130 MB, realworld experiment)")
    pd.add_argument("--perturbations", action="store_true",
                    help="also fetch the gpt_image_gen_<type>/ sets (~16 GB, aug_img / img_label)")
    pd.add_argument("--all", action="store_true", help="fetch every image group")
    pd.add_argument("--no-primary", action="store_true",
                    help="skip the main image set (fetch only the groups selected above)")
    pd.set_defaults(func=cmd_download)

    # score
    ps = sub.add_parser("score", help="score one results file or a directory")
    ps.add_argument("--input", "-i", required=True, help="results JSON file or directory")
    ps.add_argument("--output", "-o", help="output _w_metrics.json (file mode) or directory (dir mode)")
    ps.add_argument("--tail-slice", type=int, default=None,
                    help="last N chars for unknown-synonym match (reasoning outputs)")
    ps.add_argument("--score", choices=["all", "fairness", "bias", "unk"], default="all")
    ps.add_argument("--skip-existing", action="store_true",
                    help="dir mode: skip files whose output already exists")
    ps.set_defaults(func=cmd_score)

    # combine
    pc = sub.add_parser("combine", help="walk *_w_metrics.json and emit combined_metrics.json")
    pc.add_argument("--input", "-i", required=True, help="root directory")
    pc.add_argument("--output", "-o", required=True, help="output combined_metrics.json path")
    pc.set_defaults(func=cmd_combine)

    # aggregate
    pa = sub.add_parser("aggregate", help="combined_metrics.json → CSV summaries + FS/BS totals")
    pa.add_argument("--input", "-i", required=True, help="combined_metrics.json path")
    pa.add_argument("--csv-dir", required=True, help="output dir for per-category CSVs")
    pa.add_argument("--metrics-dir", required=True,
                    help="output dir for *_metrics_details.csv + totals summary")
    _add_metric_common(pa)
    pa.set_defaults(func=cmd_aggregate)

    # pipeline
    pp = sub.add_parser("pipeline", help="score → combine → aggregate in one shot")
    pp.add_argument("--input", "-i", required=True, help="results root directory")
    pp.add_argument("--output", "-o", required=True,
                    help="output directory (will contain analysis/, csv_files/, metrics_details/)")
    pp.add_argument("--tail-slice", type=int, default=None)
    _add_metric_common(pp)
    pp.set_defaults(func=cmd_pipeline)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
