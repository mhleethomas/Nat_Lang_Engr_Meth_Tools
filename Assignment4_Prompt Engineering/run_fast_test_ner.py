import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

USE_FAST_MODE = True
FAST_SENTENCE_LIMIT = 80
FULL_SENTENCE_LIMIT = None

# For local test: use small/base.  On Colab GPU uncomment large/xl for better F1.
MODEL_NAMES = [
    "google/flan-t5-small",   # 80M params
    "google/flan-t5-base",    # 250M params
    # "google/flan-t5-large", # 780M params  -- recommended on Colab T4/A100
    # "google/flan-t5-xl",    # 3B  params  -- best quality, needs ~10 GB GPU RAM
]

# Beam width for generation (higher = better structured output, slower).
NUM_BEAMS = 4

PROMPTS = {
    # --- zero-shot: plain instruction, no examples ---
    "zero_shot_plain": (
        "Named Entity Recognition task.\n"
        "Extract all named entities from the text.\n"
        "Classify each as: PER (person), ORG (organization), LOC (location), MISC (other named entity).\n"
        "The text may be in ALL CAPS. Still extract entities.\n"
        "Respond with a JSON array only. No explanation. No code.\n"
        'Format: [{"entity": "name", "type": "TYPE"}, ...]\n'
        "If no named entities exist, respond: []\n"
        "\n"
        "Text: {text}\n"
        "JSON:"
    ),
    # --- zero-shot: rubric (type definitions + strict rules) ---
    "zero_shot_rubric": (
        "You are an expert Named Entity Recognition system.\n"
        "Label every named entity in the input text using these rules:\n"
        "  PER  = person name (e.g. Barack Obama, Sampras)\n"
        "  ORG  = organization, team, or company (e.g. Reuters, FIFA, Leicestershire)\n"
        "  LOC  = location or place (e.g. London, France, Africa)\n"
        "  MISC = other named entity (e.g. nationality adjectives, event names)\n"
        "The text may be in ALL CAPS. Still identify entities.\n"
        'Output ONLY a JSON array: [{"entity": "...", "type": "..."}]. No code. No prose.\n'
        "\n"
        "Text: {text}\n"
        "JSON:"
    ),
    # --- few-shot: 3 caps-style examples matching CoNLL text format ---
    "few_shot_caps": (
        "Extract named entities. Types: PER, ORG, LOC, MISC. Return JSON array only.\n"
        "\n"
        "Text: LEICESTERSHIRE BEAT NOTTINGHAMSHIRE IN CRICKET FINAL .\n"
        'Output: [{"entity": "LEICESTERSHIRE", "type": "ORG"}, {"entity": "NOTTINGHAMSHIRE", "type": "ORG"}]\n'
        "\n"
        "Text: LONDON 1996-08-30\n"
        'Output: [{"entity": "LONDON", "type": "LOC"}]\n'
        "\n"
        "Text: UNITED STATES BEAT GERMANY 2-1 .\n"
        'Output: [{"entity": "UNITED STATES", "type": "LOC"}, {"entity": "GERMANY", "type": "LOC"}]\n'
        "\n"
        "Text: {text}\n"
        "Output:"
    ),
    # --- few-shot: balanced (includes a no-entity example) ---
    "few_shot_balanced": (
        "Extract named entities from text. Types: PER=person, ORG=organization, LOC=location, MISC=other.\n"
        "Return JSON array only.\n"
        "\n"
        "Text: THE MATCH WAS PLAYED IN PARIS .\n"
        'Output: [{"entity": "PARIS", "type": "LOC"}]\n'
        "\n"
        "Text: JOHN SMITH JOINS REUTERS IN NEW YORK .\n"
        'Output: [{"entity": "JOHN SMITH", "type": "PER"}, {"entity": "REUTERS", "type": "ORG"}, {"entity": "NEW YORK", "type": "LOC"}]\n'
        "\n"
        "Text: THE WEATHER WAS FINE YESTERDAY .\n"
        "Output: []\n"
        "\n"
        "Text: {text}\n"
        "Output:"
    ),
}

ALLOWED_TYPES = {"PER", "ORG", "LOC", "MISC"}


def read_conll(file_path: Path):
    rows = []
    tokens, tags = [], []
    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if tokens:
                    rows.append({"tokens": tokens, "tags": tags})
                    tokens, tags = [], []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                tokens.append(parts[0])
                tags.append(parts[-1])
    if tokens:
        rows.append({"tokens": tokens, "tags": tags})
    return rows


def bio_to_entities(tokens, tags):
    out = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O" or "-" not in tag:
            i += 1
            continue
        prefix, label = tag.split("-", 1)
        if prefix not in {"B", "I"}:
            i += 1
            continue
        ent = [tokens[i]]
        i += 1
        while i < len(tags) and tags[i] == f"I-{label}":
            ent.append(tokens[i])
            i += 1
        out.append({"entity": " ".join(ent), "type": label})
    return out


def build_prompt(name, text):
    return PROMPTS[name].replace("{text}", text.strip())


def parse_ner_output(raw_text):
    text = (raw_text or "").strip()
    left = text.find("[")
    right = text.rfind("]")
    candidate = text[left : right + 1] if left != -1 and right != -1 and right > left else text

    parse_ok = True
    entities = []

    try:
        loaded = json.loads(candidate)
        if isinstance(loaded, dict):
            loaded = [loaded]
        if not isinstance(loaded, list):
            loaded = []

        for item in loaded:
            if not isinstance(item, dict):
                continue
            ent = str(item.get("entity", "")).strip()
            typ = str(item.get("type", "")).strip().upper()
            if ent and typ in ALLOWED_TYPES:
                entities.append({"entity": ent, "type": typ})
    except Exception:
        parse_ok = False

    return entities, parse_ok


def to_entity_set(items):
    out = set()
    for item in items:
        ent = " ".join(str(item.get("entity", "")).lower().split())
        typ = str(item.get("type", "")).upper().strip()
        if ent and typ in ALLOWED_TYPES:
            out.add((ent, typ))
    return out


def run_fast_test():
    workspace = Path.cwd()
    assignment4_dir = workspace / "Assignment4_Prompt Engineering"
    assignment4_dir.mkdir(parents=True, exist_ok=True)

    conll_candidates = [
        Path("Assignment2_Name_Entity_Recognition/conll2003"),
        Path("../Assignment2_Name_Entity_Recognition/conll2003"),
    ]

    conll_dir = None
    for p in conll_candidates:
        if p.exists():
            conll_dir = p
            break

    if conll_dir is None:
        raise FileNotFoundError("Could not find CoNLL via relative paths")

    sentences = read_conll(conll_dir / "eng.testa")
    full_sent_count = len(sentences)

    data = []
    for s in sentences:
        text = " ".join(s["tokens"])
        gold_entities = bio_to_entities(s["tokens"], s["tags"])
        data.append({"text": text, "gold_entities": gold_entities})

    eval_df = pd.DataFrame(data)
    if USE_FAST_MODE:
        eval_df = eval_df.head(FAST_SENTENCE_LIMIT).copy()
    elif FULL_SENTENCE_LIMIT is not None:
        eval_df = eval_df.head(FULL_SENTENCE_LIMIT).copy()

    print(f"Full sentence count: {full_sent_count}")
    print(f"Fast eval sentence count: {len(eval_df)}")

    load_start = time.perf_counter()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    generators = {}
    for model_name in MODEL_NAMES:
        print(f"Loading {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
        generators[model_name] = {"tokenizer": tokenizer, "model": model, "device": device}
    load_seconds = time.perf_counter() - load_start

    metrics_start = time.perf_counter()
    all_metrics = []

    for model_name in MODEL_NAMES:
        gen = generators[model_name]
        tokenizer = gen["tokenizer"]
        model = gen["model"]
        model_device = gen["device"]
        for prompt_name in PROMPTS.keys():
            tp = fp = fn = 0
            valid = exact = 0

            for row in eval_df.itertuples(index=False):
                prompt = build_prompt(prompt_name, row.text)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
                raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                pred_entities, parse_ok = parse_ner_output(raw)

                valid += int(parse_ok)

                gold_set = to_entity_set(row.gold_entities)
                pred_set = to_entity_set(pred_entities)

                tp += len(gold_set & pred_set)
                fp += len(pred_set - gold_set)
                fn += len(gold_set - pred_set)
                exact += int(gold_set == pred_set)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

            metrics = {
                "model": model_name,
                "prompt_template": prompt_name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "valid_json_rate": round(valid / len(eval_df), 4),
                "exact_sentence_match": round(exact / len(eval_df), 4),
            }
            all_metrics.append(metrics)

    metrics_seconds = time.perf_counter() - metrics_start

    logs_start = time.perf_counter()
    log_examples = eval_df.head(min(5, len(eval_df))).copy()
    log_rows = []

    for idx, row in log_examples.iterrows():
        for model_name in MODEL_NAMES:
            gen = generators[model_name]
            tokenizer = gen["tokenizer"]
            model = gen["model"]
            model_device = gen["device"]
            for prompt_name in PROMPTS.keys():
                prompt = build_prompt(prompt_name, row["text"])
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
                raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                pred, ok = parse_ner_output(raw)
                log_rows.append(
                    {
                        "example_id": int(idx),
                        "model": model_name,
                        "prompt_template": prompt_name,
                        "parse_ok": bool(ok),
                        "text": row["text"],
                        "gold_entities": row["gold_entities"],
                        "pred_entities": pred,
                        "raw_output": raw,
                    }
                )

    logs_seconds = time.perf_counter() - logs_start
    total_seconds = load_seconds + metrics_seconds + logs_seconds

    metrics_df = pd.DataFrame(all_metrics).sort_values(by=["f1", "precision", "recall"], ascending=False).reset_index(drop=True)
    shot_df = metrics_df.copy()
    shot_df["shot_type"] = shot_df["prompt_template"].apply(lambda x: "few-shot" if x.startswith("few_shot") else "zero-shot")
    shot_df = (
        shot_df.groupby(["model", "shot_type"], as_index=False)[
            ["precision", "recall", "f1", "valid_json_rate", "exact_sentence_match"]
        ]
        .mean()
        .sort_values(by=["model", "shot_type"])
    )

    calls_metrics = len(eval_df) * len(MODEL_NAMES) * len(PROMPTS)
    calls_logs = len(log_examples) * len(MODEL_NAMES) * len(PROMPTS)

    avg_seconds_per_metric_call = metrics_seconds / calls_metrics if calls_metrics else 0.0
    estimated_full_metrics_seconds = avg_seconds_per_metric_call * full_sent_count * len(MODEL_NAMES) * len(PROMPTS)
    estimated_full_total_no_download = estimated_full_metrics_seconds + logs_seconds

    best_row = metrics_df.iloc[0].to_dict()

    summary = {
        "use_fast_mode": USE_FAST_MODE,
        "fast_sentence_limit": FAST_SENTENCE_LIMIT,
        "evaluated_sentences": int(len(eval_df)),
        "full_sentences": int(full_sent_count),
        "models": MODEL_NAMES,
        "prompts": list(PROMPTS.keys()),
        "calls_metrics": int(calls_metrics),
        "calls_logs": int(calls_logs),
        "model_load_seconds": round(load_seconds, 3),
        "metrics_seconds": round(metrics_seconds, 3),
        "logs_seconds": round(logs_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "avg_seconds_per_metric_call": round(avg_seconds_per_metric_call, 6),
        "estimated_full_metrics_seconds": round(estimated_full_metrics_seconds, 3),
        "estimated_full_total_no_download_seconds": round(estimated_full_total_no_download, 3),
        "estimated_full_total_no_download_minutes": round(estimated_full_total_no_download / 60.0, 2),
        "estimated_full_total_no_download_hours": round(estimated_full_total_no_download / 3600.0, 2),
        "best_model": best_row["model"],
        "best_prompt": best_row["prompt_template"],
        "best_f1": float(best_row["f1"]),
    }

    metrics_df.to_csv(assignment4_dir / "test_run_metrics.csv", index=False)
    shot_df.to_csv(assignment4_dir / "test_run_shot_summary.csv", index=False)
    pd.DataFrame(log_rows).to_csv(assignment4_dir / "ner_prompt_output_logs.csv", index=False)
    (assignment4_dir / "test_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== METRICS ===")
    print(metrics_df.to_string(index=False))
    print("\n=== SHOT SUMMARY ===")
    print(shot_df.to_string(index=False))
    print("\n=== TIMING SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_fast_test()
