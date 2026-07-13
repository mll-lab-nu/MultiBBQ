"""HuggingFace integration: fetch the released images for the harness.

The harness reads images from disk at each record's `image_path`
(e.g. `./images/gpt_image_gen/textual/…png`). `multibbq download` re-creates that
tree under `<root>/images/`:

    primary        the four parquet configs of MLL-Lab/MultiBBQ store every image
                   embedded; we download the parquet shards (cached under
                   ~/.cache/huggingface/) and write the original PNG bytes back to
                   images/{gpt_image_gen,imagen4ultra_image_gen}/.   (~2.7 GB)
    realworld      raw tree from MLL-Lab/MultiBBQ-realworld           (~130 MB)
    perturbations  raw tree from MLL-Lab/MultiBBQ-perturbations       (~16 GB)

The blank canvas (images/pure_white_1024_1024.png) ships with the git repository,
so it is never downloaded. Extraction is idempotent: existing files with the right
size are skipped, and an interrupted run can simply be re-run.

Requires the optional deps: `pip install "multibbq[hf]"`.
"""
import os

DEFAULT_REPO = "MLL-Lab/MultiBBQ"
REALWORLD_REPO = "MLL-Lab/MultiBBQ-realworld"
PERTURBATIONS_REPO = "MLL-Lab/MultiBBQ-perturbations"

CONFIGS = tuple(
    f"{gen}_{mod}"
    for gen in ("gpt_image_gen", "imagen4ultra_image_gen")
    for mod in ("visual_language", "visual_only")
)


def _require_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "multibbq download needs the HF extras: pip install \"multibbq[hf]\""
        ) from e


def _extract_primary(repo_id: str, root: str) -> None:
    """Download the parquet shards and write the embedded images to disk.

    We read the raw `image.bytes` column with pyarrow (no PIL decode/re-encode),
    so the extracted files are byte-identical to the released images.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    shards = [
        f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
        if f.endswith(".parquet") and f.split("/")[0] in CONFIGS
    ]
    if not shards:
        raise SystemExit(f"no parquet shards found in {repo_id} - wrong --repo?")

    written = skipped = 0
    for shard in sorted(shards):
        local = hf_hub_download(repo_id, shard, repo_type="dataset")
        table = pq.read_table(local, columns=["image", "image_path"])
        for batch in table.to_batches():
            for img, rel in zip(batch.column("image").to_pylist(),
                                batch.column("image_path").to_pylist()):
                target = os.path.join(root, rel.lstrip("./"))
                data = img["bytes"]
                if os.path.isfile(target) and os.path.getsize(target) == len(data):
                    skipped += 1
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(data)
                written += 1
        print(f"[extract] {shard}: done ({written} written, {skipped} up-to-date)")
    print(f"[primary] {written} images written, {skipped} already present")


def _snapshot_raw(repo_id: str, root: str, patterns) -> None:
    """Pull a raw image tree (repo root holds the image dirs) into <root>/images/."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id, repo_type="dataset",
        local_dir=os.path.join(root, "images"),
        allow_patterns=patterns,
    )


def download(repo_id: str = DEFAULT_REPO, root: str = ".",
             primary: bool = True, perturbations: bool = False,
             realworld: bool = False,
             realworld_repo: str = REALWORLD_REPO,
             perturbations_repo: str = PERTURBATIONS_REPO) -> int:
    """Fetch image groups into `<root>/images/`.

    Args:
        repo_id: core HuggingFace dataset id (parquet configs).
        root: directory to place the `images/` tree under (default: cwd, i.e.
            `./images/`, which is where the harness reads from).
        primary: extract the main image set from the parquet configs (default).
        perturbations: also snapshot the perturbation sets (~16 GB).
        realworld: also snapshot the real-world image set (~130 MB).
    """
    _require_hub()

    groups = [g for g, on in [("primary", primary), ("realworld", realworld),
                              ("perturbations", perturbations)] if on]
    if not groups:
        raise SystemExit("nothing selected to download")

    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    print(f"[download] groups: {', '.join(groups)} -> {os.path.join(root, 'images')}")

    if primary:
        _extract_primary(repo_id, root)
    if realworld:
        print(f"[realworld] {realworld_repo} ...")
        _snapshot_raw(realworld_repo, root, ["real_world_image/**"])
    if perturbations:
        print(f"[perturbations] {perturbations_repo} (~16 GB) ...")
        _snapshot_raw(perturbations_repo, root, ["gpt_image_gen_*/**"])

    print(f"[download] done -> {os.path.join(root, 'images')}")
    return 0
