from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import json
import os
import argparse


def run_eval(src_file, des_file, data_dir):

    with open(os.path.join(data_dir, src_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    basic_prompts = [item["prompt_basic_eval"] for item in data]
    internal_prompts = [item["prompt_internal_eval"] for item in data]
    claim_prompts = [item["prompt_claim_eval"] for item in data]

    prompt_dict = {
        "basic": basic_prompts,
        "internal": internal_prompts,
        "claim": claim_prompts
    }

    evaluator_models = {
        "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen": "Qwen/Qwen2.5-7B-Instruct"
    }


    eval_results = {
        eval_model: {ptype: [] for ptype in prompt_dict}
        for eval_model in evaluator_models
    }

    for eval_model, model_path in evaluator_models.items():
        print(f"\n🔹 Loading evaluator model: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, token=True)
        tokenizer.padding_side = tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )


        for ptype, prompts in prompt_dict.items():
            print(f" Running {ptype} prompts...")

            chat_prompts = []
            for p in prompts:
                if eval_model == "llama":
                    messages = [{"role": "user", "content": p}]
                    chat_prompts.append(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    )

                elif eval_model == "mistral":
                    chat_prompts.append(f"<s>[INST] {p} [/INST]")

                elif eval_model == "qwen":
                    chat_prompts.append(
                        "<|im_start|>user\n" + p + "<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )

            outs = pipe(
                chat_prompts,
                max_new_tokens=64,
                do_sample=False,
                batch_size=4,
                return_full_text=False,
                skip_special_tokens=True
            )

            clean = [
                (o[0] if isinstance(o, list) else o)["generated_text"].strip()
                for o in outs
            ]

            eval_results[eval_model][ptype] = clean

        print(f"Finished evaluator model: {eval_model}")


    final_data = []

    for i, item in enumerate(data):
        final_item = {
            "claim_id": item.get("claim_id"),
            "mode": item.get("mode"),
            "generator_model": item.get("model"),
            "eval_scores": {}
        }

        for ptype in prompt_dict:
            final_item["eval_scores"][ptype] = {
                eval_model: eval_results[eval_model][ptype][i]
                for eval_model in evaluator_models
            }

        final_data.append(final_item)


    save_path = os.path.join(data_dir, des_file)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved evaluation dataset → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, required=True)
    parser.add_argument("--des_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()

    run_eval(args.src_file, args.des_file, args.data_dir)
