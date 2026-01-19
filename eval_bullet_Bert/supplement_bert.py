import os
from bert_score import score
import json
import torch
import argparse
#import re


def add_bertscore_and_export_new(
    file_dir,
    file_name,
    model_type="microsoft/deberta-xlarge-mnli",
    batch_size=32
):

    file_path = os.path.join(file_dir, file_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {file_path}")

    # detect evidence fields
    evidence_keys = [k for k in data[0] if k.startswith("evidence_")]
    print("Detected evidence fields:", evidence_keys)

    pairs = []

    for idx, item in enumerate(data):

        basic_text = item.get("bullets_text_basic", "").strip()
        internal_text = item.get("bullets_text_internal", "").strip()
        claim_text = item.get("bullets_text_claim", "").strip()
        claim_field = item.get("claim", "").strip()

        if basic_text:
            # basic vs evidence_*
            for ev_key in evidence_keys:
                ev_text = item.get(ev_key, "")
                if isinstance(ev_text, list):
                    ev_text = " ".join(ev_text)
                if isinstance(ev_text, str) and ev_text.strip():
                    pairs.append((idx, "basic", ev_key, basic_text, ev_text))

            # basic vs claim
            if claim_field:
                pairs.append((idx, "basic", "claim", basic_text, claim_field))


        if internal_text:
            for ev_key in evidence_keys:
                ev_text = item.get(ev_key, "")
                if isinstance(ev_text, list):
                    ev_text = " ".join(ev_text)
                if isinstance(ev_text, str) and ev_text.strip():
                    pairs.append((idx, "internal", ev_key, internal_text, ev_text))

            # internal vs claim
            if claim_field:
                pairs.append((idx, "internal", "claim", internal_text, claim_field))


        if claim_text and claim_field:
            pairs.append((idx, "claim", "claim", claim_text, claim_field))

    print(f"Prepared {len(pairs)} BERTScore pairs.")


    all_scores = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i: i + batch_size]

        cands = [p[3] for p in batch]
        refs = [p[4] for p in batch]

        _, _, F1 = score(
            cands,
            refs,
            model_type=model_type,
            device=device,
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
        )

        all_scores.extend(F1.tolist())


    output = []
    for item in data:
        output.append({
            "claim_id": item.get("claim_id"),
            "model": item.get("model"),
            "mode": item.get("mode"),
            "bert_scores": {}
        })

    for (pair, f1) in zip(pairs, all_scores):
        idx, prompt, target_key, _, _ = pair
        output[idx]["bert_scores"].setdefault(prompt, {})
        output[idx]["bert_scores"][prompt][target_key] = float(round(f1, 4))


    out_file = file_path.replace("data_for_cal_bert_", "bert_score_")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Saved:", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, required=True)
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    add_bertscore_and_export_new(
        args.file_dir,
        args.file_name,
        args.model_type,
        args.batch_size
    )