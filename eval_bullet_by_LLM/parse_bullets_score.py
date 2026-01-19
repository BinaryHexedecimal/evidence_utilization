import re
import json
from pathlib import Path



def parse_score_string(text):

    if not text:
        return None

    def extract(pattern):
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    return {
        "criterion_1": extract(r"Criterion 1:\s*-*(-?\d+)"),
        "criterion_2": extract(r"Criterion 2:\s*-*(-?\d+)"),
        "criterion_3": extract(r"Criterion 3:\s*-*(-?\d+)"),
        "criterion_4": extract(r"Criterion 4:\s*-*(-?\d+)"),
        "relevant_bullets": extract(r"Number of relevant bullets:\s*-*(-?\d+)")
    }




def parse_eval_scores(item):
    parsed = {}

    eval_scores = item.get("eval_scores", {})
    for model_name, model_scores in eval_scores.items():
        parsed[model_name] = {}

        for mode, score_text in model_scores.items():
            parsed[model_name][mode] = parse_score_string(score_text)

    return parsed




"""
def add_parsed_bullet_score_into_result(src_dir, src_prefix, src_suffix, dst_dir):
    models = ["llama", "mistral", "qwen"]

    modes = ["2support", "2refute", "mix"]  
    for model in models:
        for mode in modes:
            
            src_filename = src_prefix + f"{model}_{mode}" + src_suffix
            src_filepath = src_dir / src_filename

            with open(src_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)   # list of items
    
            for item in data:
                item["eval_scores_parsed"] = parse_eval_scores(item)
    

            dst_filename = f"{model}_{mode}_bullet_first_result.json"
            dst_path = dst_dir / dst_filename
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

"""



def add_parsed_bullet_score_into_result(
    src_dir: Path,
    dst_dir: Path,
    src_prefix: str,
    src_suffix: str,
    dst_suffix: str,
):
    models = ["llama", "mistral", "qwen"]
    modes = ["2support", "2refute", "mix"]
    #src_prefix = "bullet_score_eval_"
    #src_suffix = "_bullet_first_result.json"

    #dst_suffix = "_bullet_first_result.json"


    for model in models:
        for mode in modes:
            # -------- source file (scores come from here) ----
            src_filename = src_prefix + f"{model}_{mode}" + src_suffix
            print(f"src_dir is {src_dir}")
            src_path = src_dir / src_filename

            with open(src_path, "r", encoding="utf-8") as f:
                src_data = json.load(f)   # list of items

            score_by_claim_id = {}
            for item in src_data:
                claim_id = item.get("claim_id")
                if claim_id is None:
                    continue
                score_by_claim_id[claim_id] = parse_eval_scores(item)

            dst_filename = f"{model}_{mode}" + dst_suffix
            dst_path = dst_dir / dst_filename

            with open(dst_path, "r", encoding="utf-8") as f:
                dst_data = json.load(f)

            # One-to-one
            missing = []
            for item in dst_data:
                cid = item.get("claim_id")
                if cid in score_by_claim_id:
                    item["eval_bullet_scores"] = score_by_claim_id[cid]
                else:
                    missing.append(cid)

            if missing:
                print(
                    f"[WARN] {model} {mode}: "
                    f"{len(missing)} claim_id(s) not matched"
                )

            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(dst_data, f, ensure_ascii=False, indent=2)

            print(f"Updated file: {dst_path}")
