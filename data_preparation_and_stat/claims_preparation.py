import pandas as pd


def group_by_claim_and_flatten_claims(src_path, dst_path):

    df = pd.read_csv(src_path, sep='\t')

    #filter relevant rows only
    df = df[df['relevant'] == True]
    print(f"After discarding Non-relevant evidence, {len(df)} evidence remain in raw data.")
   
    # group evidence by claim, separated by stance
    grouped = df.groupby(
        ['claim_id',  'claim', 'factcheck_verdict']
    ).apply(lambda g: {
        "supports": g.loc[g['evidence_stance'] == "supports", "evidence"].tolist(),
        "refutes": g.loc[g['evidence_stance'] == "refutes", "evidence"].tolist()
    },
        include_groups=False 
    ).reset_index(name="evidence_dict")



    # drop rows with no support and no refute
    grouped = grouped[grouped['evidence_dict'].apply(
        lambda d: len(d["supports"]) > 0 or len(d["refutes"]) > 0
    )]

    #determine max evidence counts
    max_supports = grouped['evidence_dict'].apply(lambda d: len(d["supports"])).max() if not grouped.empty else 0
    max_refutes = grouped['evidence_dict'].apply(lambda d: len(d["refutes"])).max() if not grouped.empty else 0

    # flatten into wide format
    flatten_data = grouped[['claim_id', 'claim', 'factcheck_verdict']].copy()

    for i in range(max_supports):
        col_name = f"support_evidence_{i+1}"
        flatten_data[col_name] = grouped['evidence_dict'].apply(
            lambda d: d["supports"][i] if i < len(d["supports"]) else ""
        )

    for i in range(max_refutes):
        col_name = f"refute_evidence_{i+1}"
        flatten_data[col_name] = grouped['evidence_dict'].apply(
            lambda d: d["refutes"][i] if i < len(d["refutes"]) else ""
        )

    print(f"Only supports and refutes evidence remain, grouped by claim_id, {len(flatten_data)} claims remains")

    #df_true.to_csv(f"{file_path}/flatten_true.tsv", sep='\t', index=False)
    flatten_data.to_csv(dst_path, sep='\t', index=False)
    print(f"Saved wide format to {dst_path}.")






# split 'druid_data.tsv' into True, False, Half True
def split_by_factcheck_verdict_and_flatten(src_path, dst_dir):
    # ------------clean---------------------
    df = pd.read_csv(src_path, sep='\t')
    print(f"Raw data has {len(df)} rows")

    # Keep only relevat evidence
    df = df[df['relevant']==True]

    # Columns to keep
    cols_to_keep = ["id", "claim_id", "claim", "evidence", "factcheck_verdict", "evidence_stance"]
    
    df = df[cols_to_keep]
    
    df = df[~df['claim_id'].str.startswith('borderlines-')]
    
    print(f"After discarding borderline claims and Non-relevant, {len(df)} evidence remain in raw data.")
    
    df = df[df["evidence_stance"].isin(["supports", "refutes"])]
    print(f"Only supports and refutes evidence remain, in total {len(df)} rows")
   
    # Group by claim_id
    grouped = df.groupby("claim_id")
    print(f"Grouped by claim_id, {len(grouped)} claims remains")
    rows = []
    max_support = 0
    max_refute = 0

    # First pass — collect evidence lists
    for cid, group in grouped:
        claim = group["claim"].iloc[0]
        supports = group[group["evidence_stance"] == "supports"]["evidence"].tolist()
        refutes = group[group["evidence_stance"] == "refutes"]["evidence"].tolist()
        verdict = group["factcheck_verdict"].iloc[0]

        # Skip claim if no evidence supports OR refutes
        if len(supports) == 0 and len(refutes) == 0:
            continue

        max_support = max(max_support, len(supports))
        max_refute = max(max_refute, len(refutes))

        rows.append({
            "claim_id": cid,
            "claim": claim,
            "supports": supports,
            "refutes": refutes,
            "factcheck_verdict": verdict
        })

    # Second pass — flatten into columns
    processed_rows = []
    for row in rows:
        flat_row = {
            "claim_id": row["claim_id"],
            "claim": row["claim"],
            "factcheck_verdict": row["factcheck_verdict"],
        }

        for i in range(max_support):
            key = f"support_evidence_{i+1}"
            flat_row[key] = row["supports"][i] if i < len(row["supports"]) else None

        for i in range(max_refute):
            key = f"refute_evidence_{i+1}"
            flat_row[key] = row["refutes"][i] if i < len(row["refutes"]) else None

        processed_rows.append(flat_row)

    df_flatten = pd.DataFrame(processed_rows)

    # group by factcheck_verdict

    df_true = df_flatten[df_flatten['factcheck_verdict'] == "True"]
    df_false = df_flatten[df_flatten['factcheck_verdict'] == "False"]
    df_half_true = df_flatten[df_flatten['factcheck_verdict'] == "Half True"]
    print("Groupped by factcheck verdict:")
    print(f"True claims: {len(df_true)}, False claims: {len(df_false)}, Half True claims: {len(df_half_true)}")

    df_true.to_csv(f"{dst_dir}/flatten_true.tsv", sep='\t', index=False)
    df_false.to_csv(f"{dst_dir}/flatten_false.tsv", sep='\t', index=False)
    df_half_true.to_csv(f"{dst_dir}/flatten_half_true.tsv", sep='\t', index=False)
    
    print("\n Done! Files are split and saved.")



