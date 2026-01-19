import os
from bert_score import score
import json
import torch
import argparse
#import re



def cal_bertscore_and_export_in_new_json(
    _dir,
    file_normal,
    file_inverse,
    model_type="microsoft/deberta-xlarge-mnli",
    batch_size=32
):

    path_A = _dir + file_normal
    path_B = _dir + file_inverse
    #out_path = Path(out_dir) / out_name
    out_file = _dir + file_inverse.replace("data_for_cal_bert_inv_ev_", "bert_inv_ev_")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("BERTScore model:", model_type)


    with open(path_A, "r", encoding="utf-8") as f:
        data_A = json.load(f)

    with open(path_B, "r", encoding="utf-8") as f:
        data_B = json.load(f)

    print(f"Loaded {len(data_A)} items from A")
    print(f"Loaded {len(data_B)} items from B")


    A_by_id = {item["claim_id"]: item for item in data_A}
    B_by_id = {item["claim_id"]: item for item in data_B}

    assert A_by_id.keys() == B_by_id.keys(), "claim_id mismatch between A and B"

    claim_ids = list(A_by_id.keys())

    prompt_types = [ "basic", "internal", "claim" ]


    merged = {ptype: {"A": [], "B": []} for ptype in prompt_types}
    meta = []

    for cid in claim_ids:
        A = A_by_id[cid]
        B = B_by_id[cid]

        meta.append({
            "claim_id": cid,
            "model": A["model"],   # generator model
            "mode": A["mode"]
        })

        for ptype in prompt_types:
            merged[ptype]["A"].append(
                merged(A[f"bullets_text_{ptype}"])
            )
            merged[ptype]["B"].append(
                merged(B[f"bullets_text_{ptype}"])
            )


    bert_results = {}

    for ptype in prompt_types:
        print(f"\nComputing BERTScore for: {ptype}")

        P, R, F1 = score(
            merged[ptype]["A"],
            merged[ptype]["B"],
            model_type=model_type,
            batch_size=batch_size,
            lang="en",
            device=device,
            verbose=True
        )

        bert_results[ptype] = {
            "precision": [p.item() for p in P],
            "recall": [r.item() for r in R],
            "f1": [f.item() for f in F1]
        }


    final_data = []

    for i, m in enumerate(meta):
        item = {
            "claim_id": m["claim_id"],
            "model": m["model"],
            "mode": m["mode"],
            "bert_model_type": model_type
        }

        for ptype in prompt_types:
            item[f"Bert_score_for_inversed_evidence_{ptype}"] = {
                "precision": bert_results[ptype]["precision"][i],
                "recall": bert_results[ptype]["recall"][i],
                "f1": bert_results[ptype]["f1"][i],
            }

        final_data.append(item)


    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved new BERTScore file → {out_file}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--_dir", type=str, required=True)
    parser.add_argument("--file_normal", type=str, required=True)
    parser.add_argument("--file_inverse", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cal_bertscore_and_export_in_new_json(
        args._dir,
        args.file_normal,
        args.file_inverse,
        model_type=args.model_type,
        batch_size=args.batch_size
    )