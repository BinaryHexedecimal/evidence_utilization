import pandas as pd

def generate_prompt_with_bullet_first(df, num_support_evidence: int, 
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
            break  # stop once we have N valid results


        support_evidence = [row[c] for c in support_cols if pd.notna(row[c]) and row[c] != ""]
        refute_evidence  = [row[c] for c in refute_cols if pd.notna(row[c]) and row[c] != ""]

        if len(support_evidence) < num_support_evidence or len(refute_evidence) < num_refute_evidence:
            continue


        claim_ids.append(row["claim_id"])
        claims.append(row["claim"])
        fact_verdicts.append(row["factcheck_verdict"])
        #print(f"append a claim_id {row['claim_id']}, claim {row['claim']}")
        

        # Select exactly required number
        selected_evidence = support_evidence[:num_support_evidence] + refute_evidence[:num_refute_evidence]

        
        if inverse:
            evidence_list = list(reversed(selected_evidence))
        else:
            evidence_list = selected_evidence


        # build basic prompt 
        prompt_lines_basic = [f"<Claim>: {row['claim']}\n"]

        for i, ev in enumerate(evidence_list):
            prompt_lines_basic.append(f"<Evidence {i+1}>: {ev}\n")

        prompt_lines_basic.append(
            "<Instruction>: Analyze the claim and the evidence above. "
            
            "Summarize the key factual insights (3–7 bullet points) that are " 
            "most relevant for evaluating the truth of the claim, " 
            "using the provided evidence. "

            "Then determine the truth of the claim "
            "based on the provided evidence. "
            
            "Respond strictly in the following format: \n"
            "Bullets: "
            "-- [point 1] "
            "-- [point 2] "
            "-- [point 3] "
            "(... up to 7 points) "
            "Internal knowledge: [true/false] "
            "Verdict: [true/false/maybe] "
            "Confidence score: [score] "

            "Rules:"  
            "1. Avoid meta-references (e.g., 'the claim states', 'the evidence shows') within the bullet points. "  
            "2. Internal knowledge refers to whether you use any internal knowledge beyond the provided evidence when producing the bullet points. "          
            
            "3. The verdict MUST be in lowercase, exactly one of: true, false, or maybe (no punctuation, no formatting, no quoting, no leading whitespace). "
            "4. Use maybe as verdict ONLY if the provided evidence contains contradiction OR is insufficient to favor either true or false. "        
            "5. The confidence score must be an integer between 1 and 100 and should reflect your certainty in the verdict. "

        )
        prompt_basic = "\n".join(prompt_lines_basic)
        results_basic.append(prompt_basic)
        


        # internal prompt
        prompt_lines_internal = [f"<Claim>: {row['claim']}\n"]

        for i, ev in enumerate(evidence_list):
            prompt_lines_internal.append(f"<Evidence {i+1}>: {ev}\n")


        prompt_lines_internal.append(
            "<Instruction>: Analyze the claim and the evidence above. "

            "Summarize the key factual insights (3–7 bullet points) that are " 
            "most relevant for evaluating the truth of the claim, " 
            "using the provided evidence and your internal knowledge. "

            "Then determine the truth of the claim "
            "based on the provided evidence and your internal knowledge. "            
            
            "Respond strictly in the following format: \n"
            "Bullets: "
            "-- [point 1] "
            "-- [point 2] "
            "-- [point 3] "
            "(... up to 7 points) "
            "Internal knowledge: [true/false] "
            "Verdict: [true/false/maybe] "
            "Confidence score: [score] "

            "Rules:"
            "1. Avoid meta-references (e.g., 'the claim states', 'the evidence shows') within the bullet points. "  
            "2. Internal knowledge refers to whether you use any internal knowledge beyond the provided evidence when producing the bullet points. "          
            
            "3. The verdict MUST be in lowercase, exactly one of: true, false, or maybe (no punctuation, no formatting, no quoting, no leading whitespace). "
            "4. Use maybe as verdict ONLY if the provided evidence contains contradiction AND your internal knowledge also fails to favor either true or false. "        
            "5. The confidence score must be an integer between 1 and 100 and should reflect your certainty in the verdict. "

        )

        prompt_internal = "\n".join(prompt_lines_internal)
        results_internal.append(prompt_internal)


        # only-claim prompt 
        prompt_lines_claim = [f"<Claim>: {row['claim']}\n"]


        prompt_lines_claim.append(
            "<Instruction>: Analyze the claim above. "
            
            "Summarize the key factual insights (3–7 bullet points) that are " 
            "most relevant for evaluating the truth of the claim. " 

            "Then determine the truth of the claim. "
            
            "Respond strictly in the following format: \n"
            "Bullets: "
            "-- [point 1] "
            "-- [point 2] "
            "-- [point 3] "
            "(... up to 7 points) "
            "Internal knowledge: [true/false] "
            "Verdict: [true/false/maybe] "
            "Confidence score: [score] "

            "Rules:"
            "1. Avoid meta-references (e.g., 'the claim states') within the bullet points. "  
            "2. Internal knowledge refers to whether you use any internal knowledge when producing the bullet points. "          
            
            "3. The verdict MUST be in lowercase, exactly one of: true, false, or maybe (no punctuation, no formatting, no quoting, no leading whitespace). "
            "4. Use maybe as verdict ONLY if your internal knowledge fails to favor either true or false. "

            "5. The confidence score must be an integer between 1 and 100 and should reflect your certainty in the verdict. "

        )

        prompt_claim = "\n".join(prompt_lines_claim)
        
        # add all three claims
        results_claim.append( prompt_claim)

    return {"claim_id": claim_ids,
            "claim":claims,
            "fact_check": fact_verdicts,
            "basic_prompt":results_basic, 
            "internal_prompt":results_internal, 
            "claim_prompt":results_claim}

