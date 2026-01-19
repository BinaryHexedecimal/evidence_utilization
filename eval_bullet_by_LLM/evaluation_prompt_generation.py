import json


def build_eval_prompt(claim, bullets):
    if isinstance(bullets, list):
        bullets_text = "\n".join(f"- {b}" for b in bullets)
    else:
        bullets_text = str(bullets)

    return (
        f"<Claim>: {claim}\n"
        f"<Bullet list>:\n{bullets_text}\n"
        "<Instruction>:\n"
        "You are an expert evaluator. "
        "Evaluate the bullet list based on the following criteria (1 = criterion is satisfied, 0 = not satisfied):\n"
        "Criterion 1: No bullet points contradict each other.\n"
        "Criterion 2: The bullet list provides sufficient information to assess the claim’s truth.\n"
        "Criterion 3: No redundancy or unnecessary repetition across bullet points.\n"
        "Criterion 4: The language is clear and easy to understand.\n"
        "Finally, evaluate how many bullet points are relevant for judging the truth of the claim.\n"
        "Rules: Output ONLY the required scores, with no explanation, no extra text, and no brackets.\n\n"
        "Respond STRICTLY in the following format, where x is 0 or 1, and y is an integer:\n"
        "Criterion 1: --x--\n"
        "Criterion 2: --x--\n"
        "Criterion 3: --x--\n"
        "Criterion 4: --x--\n"
        "Number of relevant bullets: --y--"
    )


def generate_eval_prompt_file(src_dir, src_file, dst_dir):
    with open(src_dir / src_file, encoding="utf-8") as f:
        data = json.load(f)

    new_items = []

    for item in data:
        obj = {
            "claim_id": item.get("claim_id"),
            "mode": item.get("mode"),
            "model": item.get("model"),
            "claim": item.get("claim"),

            "prompt_basic_eval": build_eval_prompt(
                item.get("claim"),
                item.get("response_basic_bullets", [])
            ),

            "prompt_internal_eval": build_eval_prompt(
                item.get("claim"),
                item.get("response_internal_bullets", [])
            ),

            "prompt_claim_eval": build_eval_prompt(
                item.get("claim"),
                item.get("response_claim_bullets", [])
            )
        }

        new_items.append(obj)


    # Build destination file name
    dst_file = (
        "prompt_for_eval_bullets_" + src_file
    ).replace("_result.json", ".json")

    # Full destination path
    dst_path = dst_dir / dst_file

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(new_items, f, ensure_ascii=False, indent=2)

    print(f"Prompts saved to: {dst_path}")





