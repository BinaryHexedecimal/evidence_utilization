import json
import re

def merge_bullets(bullets):
    if not bullets:
        return ""
    return " ".join(bullets)


def extract_tagged_section(text, tag, next_tags):
    if not text:
        return None

    next_pattern = "|".join(map(re.escape, next_tags))
    pattern = rf"{re.escape(tag)}\s*(.*?)(?={next_pattern}|$)"

    match = re.search(pattern, text, re.S)
    return match.group(1).strip() if match else None

def extract_evidence(prompt_text):
    evidence_1 = extract_tagged_section(
        prompt_text,
        "<Evidence 1>:",
        ["<Evidence 2>:", "<Instruction>:"]
    )

    evidence_2 = extract_tagged_section(
        prompt_text,
        "<Evidence 2>:",
        ["<Instruction>:"]
    )

    return evidence_1, evidence_2



def transform_json(a):
    b = {
        "model": a.get("model"),
        "mode": a.get("mode"),
        "claim_id": a.get("claim_id"),
        "claim": a.get("claim"),

        "bullets_text_basic": merge_bullets(a.get("response_basic_bullets", [])),
        "bullets_text_internal": merge_bullets(a.get("response_internal_bullets", [])),
        "bullets_text_claim": merge_bullets(a.get("response_claim_bullets", [])),
    }


    prompt_text = a.get("prompt_basic") or a.get("prompt_internal")

    evidence_1, evidence_2 = extract_evidence(prompt_text)

    b["evidence_1"] = evidence_1
    b["evidence_2"] = evidence_2

    return b





def prepare_json_for_bert(src_dir, dst_dir, src_suffix, dst_prefix, dst_suffix):
    models = ["llama", "mistral", "qwen"]
    modes = ["2support", "2refute", "mix"]

    for model in models:
        for mode in modes:
            src_filename = f"{model}_{mode}" + src_suffix
            dst_filename = dst_prefix + f"{model}_{mode}" + dst_suffix
            src_path = src_dir / src_filename
            dst_path = dst_dir / dst_filename

            with open(src_path, "r", encoding="utf-8") as f:
                json_a_list = json.load(f)   # ← this is a LIST

            json_b_list = []

            for item in json_a_list:
                json_b = transform_json(item)
                json_b_list.append(json_b)

            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(json_b_list, f, ensure_ascii=False, indent=2)