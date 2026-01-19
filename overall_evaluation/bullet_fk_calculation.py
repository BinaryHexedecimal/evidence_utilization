import json
import os
import textstat

def bullet_fk_generate(_dir, filename_suffix):

    modes = ["2support", "2refute", "mix"]
    models = ["llama", "mistral", "qwen"]

    for mode in modes:
        for model in models:

            filename = f"{model}_{mode}" + filename_suffix
            file_path = os.path.join(_dir, filename)

            print(f"\nProcessing: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:

                bullet_keys = [
                    "response_basic_bullets",
                    "response_internal_bullets",
                    "response_claim_bullets"
                ]

                for key in bullet_keys:

                    if key not in item:
                        continue

                    sentences = item[key]  
                    scored_sentences = []

                    for s in sentences:
                        grade = textstat.flesch_kincaid_grade(s)
                        scored_sentences.append({
                            "sentence": s,
                            "fk_grade": grade
                        })

                    new_key = key + "_fk"
                    item[new_key] = scored_sentences

            with open(file_path, "w", encoding="utf-8") as fw:
                json.dump(data, fw, ensure_ascii=False, indent=2)

            print(f"Updated original file: {file_path}")
