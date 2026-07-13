"""Unified MultiBBQ inference.

One prediction loop for every evaluation setting in the paper; the setting is
selected with ``--experiment`` (see :mod:`multibbq.experiments`). The loop,
prompt construction, and result schema reproduce the historical per-experiment
scripts exactly.

Usage (CLI):
    multibbq run "OpenGVLab/InternVL3_5-8B" --experiment main
    multibbq run "OpenGVLab/InternVL3_5-8B" --experiment temp --temperature 0.6
    multibbq run "gpt-5" --experiment reasoning --reasoning_mode reasoning_w_fairness
"""
import os
import glob
import json
import logging
from datetime import datetime

from tqdm import tqdm

from multibbq import utils
from multibbq.experiments import EXPERIMENTS, SYSTEM_MSGS
from multibbq.models.factory import ModelFactory


def _fields(example, cfg):
    """Select masked vs. unmasked text columns, matching the original scripts."""
    if cfg["fields"] == "unmasked":
        ambig = example["ambig_context"]
        options = example["options"]
    else:
        ambig = example["ambig_context_masked"]
        options = example["options_masked"]
    if cfg["disambig"] == "masked":
        disambig = example["disambig_context_masked"]
    else:
        disambig = example["disambig_context"]
    if cfg["inject"]:
        ambig = utils.add_in_the_image(
            ambig, example["nonstereotype_group_name"], example["stereotype_group_name"]
        )
    return ambig, disambig, options


def _resolve_image(example, cfg, args, modality):
    """Resolve the image path for the active experiment (returns '' if missing)."""
    if cfg["image"] == "none":
        return None  # text-only LLM evaluation
    if cfg["image"] == "blank":
        return "./data/images/pure_white_1024_1024.png"
    if cfg["image"] == "aug":
        # NB: perturbed images live under gpt_image_gen_<type>/ - this path rewrite is
        # gpt_image_gen-specific, faithfully reproducing the original scripts.
        return example["image_path"].replace("gpt_image_gen", f"gpt_image_gen_{args.img_aug_type}")
    if cfg["image"] == "realworld":
        category, q_id, c_id = example["category"], example["q_id"], example["c_id"]
        pattern = f"./data/images/real_world_image/{modality}_{category}_q{q_id}_c{c_id}_*.png"
        matches = glob.glob(pattern)
        return matches[0] if matches else ""
    return example["image_path"]


def _build_prompt(context, question, options, unk_label_idx, cfg):
    """Reproduce the exact prompt string for plain vs. label option styles."""
    head = f"{context}\n\n{question}\n" if context is not None else f"{question}\n"
    if cfg["options"] == "label":
        letters = ["A. ", "B. ", "C. "]
        opts = [
            letters[i] + (options[i] if unk_label_idx == i else f"person {chr(65 + i)}")
            for i in range(3)
        ]
        return head + "\n".join(opts) + "\n"
    return head + f"A. {options[0]}\nB. {options[1]}\nC. {options[2]}"


def _call_model(model, image_path, prompt, system_msg, cfg):
    """Invoke the model, with the reasoning experiments' 3-attempt retry."""
    if cfg["retry"]:
        res = None
        for _ in range(3):
            res = model.run(image_path, prompt, system_msg)
            if res is not None:
                break
        return res
    return model.run(image_path, prompt, system_msg)


def predict(data, model, cfg, args, system_msg, textual_context, ambiguity, negative):
    results = []
    for idx, example in enumerate(tqdm(data, total=len(data))):
        q_id, c_id = example["q_id"], example["c_id"]
        category = example["category"]
        modality = "visual_language" if textual_context else "visual_only"

        image_path = _resolve_image(example, cfg, args, modality)
        if cfg["image"] != "none" and (image_path == "" or not os.path.isfile(image_path)):
            print(f"Image not found: {image_path}")
            continue

        ambig, disambig, options = _fields(example, cfg)
        unk_label_idx = example["unk_label_idx"]
        question = example["neg_q"] if negative else example["nonneg_q"]
        if cfg.get("strip_image_ref"):
            question = question.replace(" in the image", "")

        if textual_context:
            context = ambig if ambiguity else ambig + " " + disambig
        else:
            context = None  # visual-only: no textual context in the prompt

        prompt = _build_prompt(context, question, options, unk_label_idx, cfg)
        res = _call_model(model, image_path, prompt, system_msg, cfg)

        if textual_context and not ambiguity:
            correct_option_idx = example["neg_label_idx"] if negative else example["nonneg_label_idx"]
        else:
            correct_option_idx = unk_label_idx

        letters = ["A.", "B.", "C."]
        logging.info("\n" + str(idx) + f"_q{q_id}_c{c_id}_" + str(image_path))
        logging.info(prompt)
        logging.info(f"pred: {res}, label: {letters[correct_option_idx]} {options[correct_option_idx]}")
        results.append({
            "image": image_path,
            "category": category,
            "options": options,
            "pred": res,
            "correct_option_idx": correct_option_idx,
            "stereotype_group_idx": example["stereotype_group_idx"],
            "nonstereotype_group_idx": example["nonstereotype_group_idx"],
            "unk_label_idx": unk_label_idx,
        })
    return results


def _str2bool(v):
    return str(v).lower() == "true"


def run(args) -> int:
    textual_context = _str2bool(args.textual_context)
    ambiguous = _str2bool(args.ambiguous)
    negative = _str2bool(args.negative)

    cfg = EXPERIMENTS[args.experiment]
    if cfg.get("text_only"):
        textual_context = True  # text-only LLM eval: the context IS the task
    if not cfg["vo"] and not textual_context:
        raise SystemExit(f"experiment '{args.experiment}' supports visual-language only")
    if cfg["image"] == "aug" and args.data_id != "gpt_image_gen":
        # The perturbation sets exist only for the GPT-Image-1 images; with any other
        # data_id the path rewrite would silently evaluate unperturbed images.
        raise SystemExit(
            f"experiment '{args.experiment}' requires --data_id gpt_image_gen "
            "(perturbed image sets exist only for the GPT-Image-1 images)")

    ambiguity_str = "ambiguous" if ambiguous else "disambiguous"
    question_type = "negative" if negative else "nonnegative"
    # text-only LLM runs are tagged 'text' (no visual modality) so they are not
    # conflated with the multimodal visual_language results.
    if cfg.get("text_only"):
        modality = "text"
    else:
        modality = "visual_language" if textual_context else "visual_only"

    # --- resolve model mode + system message + result token/suffix ---
    mode = cfg["mode"]
    system_msg = SYSTEM_MSGS["nonreasoning"]
    file_suffix = ""
    token = cfg["token"]
    if args.experiment == "reasoning":
        rm = args.reasoning_mode
        mode = "default" if "nonreasoning" in rm else "reasoning"
        system_msg = SYSTEM_MSGS[rm]
        token = rm
        if rm != "nonreasoning":
            file_suffix = f"_{rm}"
    elif args.experiment == "temp":
        token = f"temp_{args.temperature}"
    elif cfg["image"] == "aug":
        token = args.img_aug_type

    model = ModelFactory().create_model(
        args.model_id, mode=mode, quant=cfg["quant"],
        temperature=args.temperature if mode == "temp" else None,
        text_only=cfg.get("text_only", False),
    )

    file_name = f"{model.name}_{model.size}_{modality}_{question_type}_{ambiguity_str}{file_suffix}"
    print(f"[run] {args.model_id} -> {file_name}")

    out_dir = f"./results/{args.data_id}_{token}"
    formatted_time = datetime.now().strftime("%m%d%H%M")
    log_filepath = f"{out_dir}/{args.model_id}_logs/{file_name}_{formatted_time}.log"
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="", filename=log_filepath)

    data_file = "multibbq_visual_language.json" if textual_context else "multibbq_visual_only.json"
    with open(f"./data/metadata/{args.data_id}/{data_file}") as f:
        data = json.load(f)

    results = predict(data, model, cfg, args, system_msg,
                      textual_context, ambiguous, negative)
    if not results:
        print(f"[warn] 0 of {len(data)} records evaluated; nothing usable was found "
              "(are the images downloaded? see docs/getting-started/installation.md)")

    pred_datapath = f"{out_dir}/{args.model_id}/{file_name}.json"
    os.makedirs(os.path.dirname(pred_datapath), exist_ok=True)
    with open(pred_datapath, "w") as f:
        json.dump({"data": results}, f)
    return 0
