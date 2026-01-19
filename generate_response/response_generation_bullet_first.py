import argparse
import json
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)


from prompts_generation_bullet_first import generate_prompt_with_bullet_first


def generate_bullet_reply(
    src_file: str,
    des_file: str,
    des_dir: str,
    num_support_evidence: int,
    num_refute_evidence: int,
    mode: str,
    N: int = None,
    inverse: bool = False,
):
    df = pd.read_csv(src_file, sep="\t")

    prompt_dict = generate_prompt_with_bullet_first(
        df, num_support_evidence, num_refute_evidence, N, inverse
    )

    if len(prompt_dict["claim_id"]) == 0:
        print("No items to process.")
        return

    models_map = {
        "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
    }

    #candidate_tokens = ["true", "false", "maybe"]

    for model_name, model_path in models_map.items():
        print(f"\n=== Loading model: {model_path} ===")

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=bnb,
        )
        model.eval()

        types = ["basic", "internal", "claim"]

        for pt in types:
            prompts = prompt_dict[f"{pt}_prompt"]

            # chat-templates
            chat_prompts = [build_chat_prompt(model_name, p, tokenizer) for p in prompts]

            # text generation (old reliable code)
            generated_texts = generate_text_pipeline(
                model, tokenizer, chat_prompts, batch_size=4
            )

            # store
            prompt_dict[f"gen_{pt}"] = generated_texts
           
        # assemble results
        final = []
        for i in range(len(prompt_dict["claim_id"])):
            row = {
                "model": model_name,
                "mode": mode,
                "claim_id": prompt_dict["claim_id"][i],
                "claim": prompt_dict["claim"][i],
                "factcheck_verdict": prompt_dict["fact_check"][i],

                "prompt_basic": prompt_dict["basic_prompt"][i],
                "prompt_internal": prompt_dict["internal_prompt"][i],
                "prompt_claim": prompt_dict["claim_prompt"][i],

                "response_basic": prompt_dict["gen_basic"][i],
                "response_internal": prompt_dict["gen_internal"][i],
                "response_claim": prompt_dict["gen_claim"][i],

                # "logits_basic": prompt_dict["logits_basic"][i],
                # "logits_internal": prompt_dict["logits_internal"][i],
                # "logits_claim": prompt_dict["logits_claim"][i],
            }
            final.append(row)

        os.makedirs(des_dir, exist_ok=True)
        save_path = os.path.join(des_dir, f"{model_name}_{mode}_{des_file}")
        if inverse:
            save_path = os.path.join(des_dir, f"{model_name}_{mode}_inverse_{des_file}")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(final)} items to {save_path}")



def build_chat_prompt(model_name, prompt, tokenizer):
    name = model_name.lower()

    if "llama" in name:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    if "mistral" in name:
        return (
            f"<s>[INST] <<SYS>>\n"
            f"You are a helpful assistant.\n"
            f"<</SYS>>\n\n"
            f"{prompt} [/INST]"
        )

    if "qwen" in name:
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    raise ValueError("Unknown model type.")


def generate_text_pipeline(model, tokenizer, chat_prompts, batch_size=4):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        device_map="auto",
    )

    outputs = pipe(
        chat_prompts,
        max_new_tokens=512,
        do_sample=False,
        #repetition_penalty=1.0,
        skip_special_tokens=True,
        batch_size=batch_size,
    )

    return [o[0]["generated_text"] for o in outputs]





# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, required=True)
    parser.add_argument("--des_file", type=str, required=True)
    parser.add_argument("--des_dir", type=str, required=True)
    parser.add_argument("--num_support_evidence", type=int, required=True)
    parser.add_argument("--num_refute_evidence", type=int, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--inverse", action="store_true", help="Reverse the evidence order")
    #parser.add_argument("--gen_batch_size", type=int, default=4, help="Micro-batch size for generation")
    args = parser.parse_args()

    generate_bullet_reply(
        src_file=args.src_file,
        des_file=args.des_file,
        des_dir=args.des_dir,
        num_support_evidence=args.num_support_evidence,
        num_refute_evidence=args.num_refute_evidence,
        mode=args.mode,
        N=args.N,
        inverse=args.inverse,
        #generation_batch_size=args.gen_batch_size,
    )
