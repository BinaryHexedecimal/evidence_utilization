import argparse
import json
#import math
import os
#from typing import List, Dict
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    #pipeline,
    BitsAndBytesConfig,
)


from prompts_generation_only_verdict import generate_prompt_with_only_verdict
from prompts_generation_verdict_first import generate_prompt_with_verdict_first
from prompts_generation_verdict_confidence import generate_prompt_with_verdict_confidence   



def generate_bullet_reply(
    src_file: str,
    des_file: str,
    des_dir: str,
    num_support_evidence: int,
    num_refute_evidence: int,
    mode: str,
    verdict_mode: str,
    N: int = None,
    inverse: bool = False,
):
    df = pd.read_csv(src_file, sep="\t")

    if verdict_mode == "verdict_first":
        prompt_dict = generate_prompt_with_verdict_first(
            df, num_support_evidence, num_refute_evidence, N, inverse
        )
    elif verdict_mode == "verdict_only":
        prompt_dict = generate_prompt_with_only_verdict(
            df, num_support_evidence, num_refute_evidence, N, inverse
        )
    elif verdict_mode == "verdict_confidence":
        prompt_dict = generate_prompt_with_verdict_confidence(
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

    candidate_tokens = ["true", "false", "maybe"]

    for model_name, model_path in models_map.items():
        print(f"\n=== Loading model: {model_path} ===")

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)


        # force first token to be true/false/maybe
        verdict_tokens = ["true", "false", "maybe"]
        forced_token_ids = [
            tokenizer(tok, add_special_tokens=False).input_ids[0]
            for tok in verdict_tokens
        ]


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
                model, tokenizer, chat_prompts, forced_token_ids, batch_size=4,
            )

            # logits (separate small forward pass)
            logits, raw_logits_list = compute_first_token_logits(
                model, tokenizer, chat_prompts, candidate_tokens,forced_token_ids, micro_batch=4
            )


            for idx, (gen, logit_dict, full_logits) in enumerate(zip(generated_texts, logits, raw_logits_list)):

                predicted = max(logit_dict, key=logit_dict.get)

                actual = gen.strip().split()[0] if gen.strip() else "<EMPTY>"

                global_id = int(torch.argmax(full_logits))
                global_tok = tokenizer.convert_ids_to_tokens(global_id)

                print(f"\n[DEBUG] Item {idx} ({pt} prompt)")
                print("  Candidate-best verdict:", predicted)
                print("  Actual first token:    ", actual)
                print("  Global top-1 token:    ", repr(global_tok))
                print("  Global top-1 ID:       ", global_id)
                print("  Candidate logits:      ", logit_dict)
                print("  Output snippet:        ", gen[:120].replace('\n', '\\n'))

            # store
            prompt_dict[f"gen_{pt}"] = generated_texts
            prompt_dict[f"logits_{pt}"] = logits

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

                "logits_basic": prompt_dict["logits_basic"][i],
                "logits_internal": prompt_dict["logits_internal"][i],
                "logits_claim": prompt_dict["logits_claim"][i],
            }
            final.append(row)

        os.makedirs(des_dir, exist_ok=True)
        save_path = os.path.join(des_dir, f"{model_name}_{mode}_{des_file}")
        if inverse:
            save_path = os.path.join(des_dir, f"inverse_ev_{model_name}_{mode}_{des_file}")

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



def generate_text_pipeline(model, tokenizer, chat_prompts, forced_token_ids, batch_size=4):
    generated_texts = []
    full_vocab = list(range(model.config.vocab_size))
    for i in range(0, len(chat_prompts), batch_size):
        batch = chat_prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)

        prompt_len = enc.input_ids.shape[1]

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            
            cur_len = input_ids.shape[-1]

            if cur_len == prompt_len:
                return forced_token_ids
            else:
                return full_vocab  

        outputs = model.generate(
            **enc,
            max_new_tokens=512,
            do_sample=False,
            #repetition_penalty=1.1,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,  
        )

        decoded = tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )
        generated_texts.extend(decoded)

    return generated_texts

def compute_first_token_logits(model, tokenizer, chat_prompts, candidate_tokens, forced_token_ids, micro_batch=4):
#def compute_first_token_logits(model, tokenizer, chat_prompts, candidate_tokens, micro_batch=4):
    cand_ids = {
        tok: tokenizer(tok, add_special_tokens=False).input_ids[0]
        for tok in candidate_tokens
    }

    candidate_logits = []
    raw_logits = []
    full_vocab = list(range(model.config.vocab_size))

    for i in range(0, len(chat_prompts), micro_batch):
        batch = chat_prompts[i:i + micro_batch]

        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():

            prompt_len = enc.input_ids.shape[1]

            def prefix_allowed_tokens_fn(batch_id, input_ids):
  
                cur_len = input_ids.shape[-1]
                

                if cur_len == prompt_len:
                    return forced_token_ids
                else:
                    return full_vocab

            out = model.generate(
                **enc,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn  
            )


        step_logits = out.scores[0]  

        for row in step_logits:
            raw_logits.append(row.cpu())
            cand = {tok: float(row[cid]) for tok, cid in cand_ids.items()}
            candidate_logits.append(cand)

    return candidate_logits, raw_logits





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", type=str, required=True)
    parser.add_argument("--des_file", type=str, required=True)
    parser.add_argument("--des_dir", type=str, required=True)
    parser.add_argument("--num_support_evidence", type=int, required=True)
    parser.add_argument("--num_refute_evidence", type=int, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--verdict_mode", type=str, choices=["verdict_only", "verdict_first", "verdict_confidence"], required=True)
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
        verdict_mode=args.verdict_mode,
        N=args.N,
        inverse=args.inverse,
        #generation_batch_size=args.gen_batch_size,
    )
