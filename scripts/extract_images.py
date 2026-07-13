#!/usr/bin/env python3
"""Download the MultiBBQ parquet configs and extract the main image set.

Standalone: needs only `pip install huggingface_hub pyarrow` (NOT the multibbq
package). It mirrors `_extract_primary` in `multibbq/hf.py`, which is what
`multibbq download` runs; use this when you want the image tree without
installing the toolkit.

    python scripts/extract_images.py                 # -> ./data/images/  (~2.7 GB)
    python scripts/extract_images.py --root /path    # -> /path/data/images/

The parquet shards land in the standard HF cache (~/.cache/huggingface/), so a
repo you already pulled with `hf download MLL-Lab/MultiBBQ --repo-type dataset`
is not downloaded again. Extraction writes each row's embedded PNG back to its
`image_path` (raw bytes, no re-encode: files are byte-identical to the released
images) and is idempotent - re-run it to resume.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="MLL-Lab/MultiBBQ", help="HuggingFace dataset id")
    ap.add_argument("--root", default=".",
                    help="directory to place the data/images/ tree under (default: cwd)")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    configs = {f"{gen}_{mod}"
               for gen in ("gpt_image_gen", "imagen4ultra_image_gen")
               for mod in ("visual_language", "visual_only")}
    shards = sorted(f for f in HfApi().list_repo_files(args.repo, repo_type="dataset")
                    if f.endswith(".parquet") and f.split("/")[0] in configs)
    if not shards:
        raise SystemExit(f"no parquet shards found in {args.repo}")

    written = skipped = 0
    for shard in shards:
        local = hf_hub_download(args.repo, shard, repo_type="dataset")
        table = pq.read_table(local, columns=["image", "image_path"])
        for batch in table.to_batches():
            for img, rel in zip(batch.column("image").to_pylist(),
                                batch.column("image_path").to_pylist()):
                target = os.path.join(args.root, rel.lstrip("./"))
                data = img["bytes"]
                if os.path.isfile(target) and os.path.getsize(target) == len(data):
                    skipped += 1
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(data)
                written += 1
        print(f"[extract] {shard}: done ({written} written, {skipped} up-to-date)")
    print(f"[done] {written} images written, {skipped} already present "
          f"-> {os.path.join(args.root, 'data', 'images')}")


if __name__ == "__main__":
    main()
