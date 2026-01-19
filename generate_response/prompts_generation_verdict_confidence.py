import pandas as pd

def generate_prompt_with_verdict_confidence(df, num_support_evidence: int, 
                      num_refute_evidence: int, N: int = None, 
                      inverse: bool = False):

    support_cols = [col for col in df.columns if col.startswith("support_evidence")]
    refute_cols = [col for col in df.columns if col.startswith("refute_evidence")]

    claim_ids = []
    claims = []
    fact_verdicts = []
    results_basic = []
    results_internal = []
    results_claim = []

    for _, row in df.iterrows():

        if N is not None and len(claim_ids) > N:
            break  


        support_evidence = [row[c] for c in support_cols if pd.notna(row[c]) and row[c] != ""]
        refute_evidence  = [row[c] for c in refute_cols if pd.notna(row[c]) and row[c] != ""]


        if len(support_evidence) < num_support_evidence or len(refute_evidence) < num_refute_evidence:
            continue


        claim_ids.append(row["claim_id"])
        claims.append(row["claim"])
        fact_verdicts.append(row["factcheck_verdict"])

        selected_evidence = support_evidence[:num_support_evidence] + refute_evidence[:num_refute_evidence]

        
        if inverse:
            evidence_list = list(reversed(selected_evidence))
        else:
            evidence_list = selected_evidence



        prompt_lines_basic = [f"<Claim>: {row['claim']}\n"]

        for i, ev in enumerate(evidence_list):
            prompt_lines_basic.append(f"<Evidence {i+1}>: {ev}\n")

        prompt_lines_basic.append(
            "<Instruction>: Analyze the claim and the evidence above. "
            "Determine the truth of the claim "
            "based on the provided evidence. "
            
            "Respond strictly in the following format: \n"
            "[true/false/maybe] "
            "Confidence score: [score] "


            "Rules:"          
            "1. Your response MUST begin with the verdict as the first token in lowercase, exactly one of: true, false, or maybe (no punctuation, no formatting, no quoting, no leading whitespace). "
            "2. Use maybe as verdict ONLY if the provided evidence contains contradiction OR is insufficient to favor either true or false. "        
            "3. The confidence score must be an integer between 1 and 100 and should reflect your certainty in the verdict. "

        )
        prompt_basic = "\n".join(prompt_lines_basic)
        results_basic.append(prompt_basic)
        


        prompt_lines_internal = [f"<Claim>: {row['claim']}\n"]

        for i, ev in enumerate(evidence_list):
            prompt_lines_internal.append(f"<Evidence {i+1}>: {ev}\n")


        prompt_lines_internal.append(
            "<Instruction>: Analyze the claim and the evidence above. "
            "Determine the truth of the claim "
            "based on the provided evidence and your internal knowledge. "
            
            "Respond strictly in the following format: \n"
            "[true/false/maybe] "
            "Confidence score: [score] "

            "Rules:"
            "1. Your response MUST begin with the verdict as the first token in lowercase, exactly one of: true, false, or maybe (no punctuation, no formatting, no quoting, no leading whitespace). "
            "2. Use maybe as verdict ONLY if the provided evidence contains contradiction AND your internal knowledge also fails to favor either true or false. "
            "3. The confidence score must be an integer between 1 and 100 and should reflect your certainty in the verdict. "
            
        )

        prompt_internal = "\n".join(prompt_lines_internal)
        results_internal.append(prompt_internal)


        prompt_lines_claim = [f"<Claim>: {row['claim']}\n"]


        prompt_lines_claim.append(
            "<Instruction>: Analyze the claim above. "
            "Determine the truth of the claim. "
            
            "Respond strictly in the following format: \n"
            "[true/false/maybe] "
            "Confidence score: [score] "

            "Rules:"
            "1. Your response MUST begin with the verdict as the first token in lowercase, exactly one of: true, false, or maybe (no punctuation, no formatting, no quoting, no leading whitespace)."
            "2. Use maybe as verdict ONLY if your internal knowledge fails to favor either true or false. "
            "3. The confidence score must be an integer between 1 and 100 and should reflect your certainty in the verdict. "
            
        )

        prompt_claim = "\n".join(prompt_lines_claim)
        

        results_claim.append( prompt_claim)

    return {"claim_id": claim_ids,
            "claim":claims,
            "fact_check": fact_verdicts,
            "basic_prompt":results_basic, 
            "internal_prompt":results_internal, 
            "claim_prompt":results_claim}

