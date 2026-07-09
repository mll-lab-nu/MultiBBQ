"""HuggingFace integration: fetch the released images for the harness.

The MultiBBQ dataset on the Hub stores images as a raw tree that already matches
the layout the inference harness expects (each record's `image_path`, e.g.
`./images/gpt_image_gen/textual/…png`). `multibbq download` pulls that tree into place.

Groups (all under `images/` on the Hub):
    primary        gpt_image_gen/, imagen4ultra_image_gen/           (~2.7 GB) baseline experiments
    perturbations  gpt_image_gen_<type>/                   (~16 GB)  aug_img / img_label
    realworld      real_world_image/               (~0.2 GB) realworld
    blank          pure_white_1024_1024.png                  unmasked_wo_img

Requires the optional deps: `pip install "multibbq[hf]"` (huggingface_hub).
"""
import os

DEFAULT_REPO = "MLL-Lab/MultiBBQ"


def _patterns(primary, perturbations, realworld, blank):
    pats = []
    if primary:
        pats += ["images/gpt_image_gen/**", "images/imagen4ultra_image_gen/**"]
    if perturbations:
        pats += ["images/gpt_image_gen_*/**"]
    if realworld:
        pats += ["images/real_world_image/**"]
    if blank:
        pats += ["images/pure_white_1024_1024.png"]
    return pats


def download(repo_id: str = DEFAULT_REPO, root: str = ".",
             primary: bool = True, perturbations: bool = True,
             realworld: bool = True, blank: bool = True) -> int:
    """Download image groups from the Hub into `<root>/images/`.

    Args:
        repo_id: HuggingFace dataset id.
        root: directory to place the `images/` tree under (default: cwd, i.e.
            `./images/`, which is where the harness reads from).
        primary/perturbations/realworld/blank: which image groups to fetch.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "multibbq download needs the HF extras: pip install \"multibbq[hf]\""
        ) from e

    patterns = _patterns(primary, perturbations, realworld, blank)
    if not patterns:
        raise SystemExit("nothing selected to download")

    os.makedirs(root, exist_ok=True)
    print(f"[download] {repo_id} -> {os.path.join(root, 'images')} "
          f"(groups: {', '.join(g for g, on in [('primary', primary), ('perturbations', perturbations), ('realworld', realworld), ('blank', blank)] if on)})")
    snapshot_download(
        repo_id, repo_type="dataset", local_dir=root,
        allow_patterns=patterns,
    )
    print(f"[download] done -> {os.path.join(root, 'images')}")
    return 0
