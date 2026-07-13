#!/usr/bin/env python3
"""Build and push the MultiBBQ HuggingFace datasets.

Four repos under the MLL-Lab org, one per artifact:
  MLL-Lab/MultiBBQ                pure parquet; 4 configs
                                  ({gpt_image_gen,imagen4ultra_image_gen}_{visual_language,visual_only})
                                  with embedded images. `load_dataset`-able, Hub viewer works.
                                  `multibbq download` re-creates the raw ./data/images/ tree from it.
  MLL-Lab/MultiBBQ-realworld      the raw real_world_image/ set (~130 MB).
  MLL-Lab/MultiBBQ-perturbations  the raw gpt_image_gen_<type>/ perturbation sets (~16 GB).
  MLL-Lab/MultiBBQ-results        raw model outputs + computed metrics (uploaded separately).

Reads metadata from <source>/data (the repository's canonical data/) and images from
<source>/data/images. Authenticate first: `huggingface-cli login`, or set HF_TOKEN in the
environment.

    pip install "multibbq[hf]"
    python scripts/build_hf_dataset.py --source ../hf_source            # build + verify only
    python scripts/build_hf_dataset.py --source ../hf_source --push     # build + push
"""
import argparse
import json
import os

GENERATORS = ("gpt_image_gen", "imagen4ultra_image_gen")
MODALITIES = ("visual_language", "visual_only")


SHARD_TARGET_BYTES = 450_000_000  # ~450 MB/shard, matching push_to_hub defaults


def stage_config(ds, stage_dir, config, split="test", shard_target=SHARD_TARGET_BYTES):
    """Embed image bytes and write sharded parquet under <stage_dir>/<config>/.

    We deliberately avoid `Dataset.push_to_hub` (its post-upload card-merge reads
    the remote README through fsspec, which intermittently returns a corrupted
    body and raises a spurious UnicodeDecodeError). Instead we build parquet
    locally and upload the tree with `upload_folder`, which never touches that
    code path.

    `Dataset.to_parquet` on a path-based Image column would store only the file
    path, not the bytes, so the Hub viewer would break. We embed explicitly with
    `embed_table_storage` first, exactly what push_to_hub(embed_external_files=True)
    does internally.
    """
    import math
    from datasets.table import embed_table_storage

    ds_e = ds.with_format("arrow").map(embed_table_storage, batched=True, keep_in_memory=True)
    n = max(1, math.ceil(ds_e.data.nbytes / shard_target))
    cfg_dir = os.path.join(stage_dir, config)
    os.makedirs(cfg_dir, exist_ok=True)
    for i in range(n):
        shard = ds_e.shard(num_shards=n, index=i, contiguous=True)
        fname = f"{split}-{i:05d}-of-{n:05d}.parquet"
        shard.to_parquet(os.path.join(cfg_dir, fname))
    print(f"        staged {n} shard(s) -> {cfg_dir}")
    return n


def build_config(source: str, generator: str, modality: str):
    from datasets import Dataset, Image

    rows = json.load(open(f"{source}/data/{generator}/multibbq_{modality}.json"))
    kept, missing = [], 0
    for r in rows:
        path = os.path.join(source, r["image_path"].lstrip("./"))
        if os.path.isfile(path):
            kept.append({**r, "image": path})
        else:
            missing += 1
    if missing:
        print(f"[warn] {generator}/{modality}: {missing}/{len(rows)} images missing locally")

    ds = Dataset.from_list(kept).cast_column("image", Image())
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=".", help="dir containing data/ (with data/images/)")
    ap.add_argument("--repo", default="MLL-Lab/MultiBBQ")
    ap.add_argument("--realworld-repo", default="MLL-Lab/MultiBBQ-realworld")
    ap.add_argument("--perturbations-repo", default="MLL-Lab/MultiBBQ-perturbations")
    ap.add_argument("--split", default="test")
    ap.add_argument("--stage-dir", default="hf_stage",
                    help="local dir to assemble the core repo tree before upload")
    ap.add_argument("--push", action="store_true", help="upload to the Hub (else stage + verify only)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--skip-core", action="store_true")
    ap.add_argument("--skip-realworld", action="store_true")
    ap.add_argument("--skip-perturbations", action="store_true")
    args = ap.parse_args()

    api = None
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()  # token from `huggingface-cli login` or HF_TOKEN

    # --- core repo: pure parquet (4 configs) + card ---
    # <stage>/README.md, <stage>/<config>/test-*.parquet
    if not args.skip_core:
        stage = args.stage_dir
        os.makedirs(stage, exist_ok=True)

        for gen in GENERATORS:
            for mod in MODALITIES:
                config = f"{gen}_{mod}"
                print(f"[build] {config} ...")
                ds = build_config(args.source, gen, mod)
                print(f"        {len(ds)} rows")
                stage_config(ds, stage, config, split=args.split)

        # our hand-written card (declares all four configs' data_files, so the viewer works)
        card = "docs/huggingface/hf_dataset_card.md"
        if os.path.isfile(card):
            import shutil
            shutil.copyfile(card, os.path.join(stage, "README.md"))
            print("[card] README.md")

        # upload the assembled tree in one shot (no push_to_hub, no card-merge)
        if args.push:
            api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
            api.upload_folder(folder_path=stage, repo_id=args.repo, repo_type="dataset",
                              commit_message="Rebuild parquet configs from canonical data/")
            print(f"[upload] {stage} -> {args.repo}")

    # --- real-world images: separate repo, raw file tree ---
    if not args.skip_realworld:
        rw = os.path.join(args.source, "data", "images", "real_world_image")
        n = len(os.listdir(rw)) if os.path.isdir(rw) else 0
        print(f"[realworld] {n} files in {rw}")
        if args.push and n:
            api.create_repo(args.realworld_repo, repo_type="dataset",
                            private=args.private, exist_ok=True)
            api.upload_folder(folder_path=rw, path_in_repo="real_world_image",
                              repo_id=args.realworld_repo, repo_type="dataset",
                              commit_message="Add real-world image set")
            card = "docs/huggingface/hf_realworld_card.md"
            if os.path.isfile(card):
                api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                                repo_id=args.realworld_repo, repo_type="dataset")
            print(f"[upload] real_world_image -> {args.realworld_repo}")

    # --- perturbations: separate repo, raw file tree ---
    if not args.skip_perturbations:
        pert = sorted(d for d in os.listdir(os.path.join(args.source, "data", "images"))
                      if d.startswith("gpt_image_gen_"))
        print(f"[perturbations] {len(pert)} sets: {pert}")
        if args.push and pert:
            api.create_repo(args.perturbations_repo, repo_type="dataset",
                            private=args.private, exist_ok=True)
            for d in pert:
                api.upload_folder(folder_path=os.path.join(args.source, "data", "images", d),
                                  path_in_repo=d, repo_id=args.perturbations_repo,
                                  repo_type="dataset")
                print(f"[upload] {d} -> {args.perturbations_repo}")

    if args.push:
        print(f"[done] https://huggingface.co/datasets/{args.repo}")
    else:
        print("[done] build/verify only (pass --push to upload)")


if __name__ == "__main__":
    main()
