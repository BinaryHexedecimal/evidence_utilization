import pandas as pd
import json
import os
#import glob
import re
#import itertools
import numpy as np
import matplotlib.pyplot as plt




def plot_verdict_per_mode(file_dir, file_list, save_suffix, factcheck= ""):
    
    records = []
    for file in file_list:
        file_path = file_dir + file
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                model = item["model"]
                mode = item["mode"]
                for prompt in ["basic", "internal", "claim"]:
                    records.append({
                        "model": model,
                        "mode": mode,
                        "prompt": prompt,
                        "verdict": item.get(f"response_{prompt}_verdict", None)
                    })
    
    df = pd.DataFrame(records)


    df['verdict'] = df['verdict'].fillna("missing").str.strip().str.lower()

    # keep only allowed verdicts
    valid_verdicts = {"true", "false", "maybe"}
    df.loc[~df['verdict'].isin(valid_verdicts), 'verdict'] = "missing"

    # consistent ordering
    models = ["llama", "mistral", "qwen"]
    prompts = ["basic", "internal", "claim"]

    
    colors = {
        "true": "#4daf4a",
        "false": "#e41a1c",
        "maybe": "#ffd92f",
        "missing": "gray"
    }
    

    for mode in sorted(df['mode'].unique()):
        df_mode = df[df['mode'] == mode]

        # number of unique claims (or total rows / 3)
        n_items = int(len(df_mode) / 9) # df_mode['claim_id'].nunique() #or 

        print(len(df_mode))

    
        grouped = df_mode.groupby(['model', 'prompt', 'verdict']).size().unstack(fill_value=0)
        grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    
        for v in ["true", "false", "maybe", "missing"]:
            if v not in grouped_pct.columns:
                grouped_pct[v] = 0
        grouped_pct = grouped_pct[["true", "false", "maybe", "missing"]]
    

        fig, ax = plt.subplots(figsize=(5,5))
    
        # X positions: 3 bars per model, spaced groups
        group_gap = 0.4  # space between model groups
        bar_width = 0.2  # narrower bars
        inside_gap = 0.05
    
        x_positions = []
        labels = []
        current_x = 0
        for model in models:
            for prompt in prompts:
                x_positions.append(current_x)
                labels.append(f"{model}, {prompt}")
                current_x += bar_width + inside_gap
            current_x += group_gap  # extra gap between models
    

        plot_data = []
        for model in models:
            for prompt in prompts:
                if (model, prompt) in grouped_pct.index:
                    plot_data.append(grouped_pct.loc[(model, prompt)])
                else:
                    plot_data.append(pd.Series({v: 0 for v in ["true", "false", "maybe", "missing"]}))
    
        plot_df = pd.DataFrame(plot_data, index=labels)
    

        bottom = np.zeros(len(plot_df))
        for verdict in ["true", "false", "maybe", "missing"]:
            ax.bar(x_positions, plot_df[verdict], bottom=bottom, width=bar_width, label=verdict, color=colors[verdict])
            bottom += plot_df[verdict].values
    

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Percentage of Verdicts (%)")
        #ax.set_title(f"{mode}: Verdict Distribution")
        if len(factcheck) != 0:
            ax.set_title(f"verdicts (n = {n_items}): {factcheck}, {mode}")
        else:
             ax.set_title(f"verdicts (n = {n_items}): {mode}")
        ax.set_ylim(0, 100)
        
        ax.legend(title="Verdict")
        plt.tight_layout()
        if len(factcheck) != 0:
            plt.savefig(f"{file_dir}figures/verdict_{factcheck}_{mode}_{save_suffix}.png", dpi=300)
        else:
            plt.savefig(f"{file_dir}figures/verdict_{mode}_{save_suffix}.png", dpi=300)

        plt.show()









def plot_internal_per_mode(file_dir, file_list, save_suffix, factcheck="" ):

    records = []
    for file in file_list:
        file_path = file_dir + file
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                model = item["model"]
                mode = item["mode"]
                for prompt in ["basic", "internal", "claim"]:
                    records.append({
                        "model": model,
                        "mode": mode,
                        "prompt": prompt,
                        "internal_used": item.get(f"response_{prompt}_internal_knowledge", None)
                    })


    df = pd.DataFrame(records)
    df['internal_used'] = df['internal_used'].fillna("missing").str.strip().str.lower()


    valid_answers = {"true", "false"}
    df.loc[~df['internal_used'].isin(valid_answers), 'internal_used'] = "missing"



    models = ["llama", "mistral", "qwen"]
    prompts = ["basic", "internal", "claim"]


    colors = {
        "true": "#377eb8",       # blue
        "false": "#e41a1c",      # red
        "missing": "gray"        # unknown
    }


    for mode in sorted(df['mode'].unique()):
        df_mode = df[df['mode'] == mode]


        n_items = int(len(df_mode) / 9)

        grouped = df_mode.groupby(['model', 'prompt', 'internal_used']).size().unstack(fill_value=0)
        grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

        for v in ["true", "false", "missing"]:
            if v not in grouped_pct.columns:
                grouped_pct[v] = 0
        grouped_pct = grouped_pct[["true", "false", "missing"]]

        fig, ax = plt.subplots(figsize=(5, 5))

        group_gap = 0.4
        bar_width = 0.2
        inside_gap = 0.05

        x_positions = []
        labels = []
        current_x = 0
        for model in models:
            for prompt in prompts:
                x_positions.append(current_x)
                labels.append(f"{model}, {prompt}")
                current_x += bar_width + inside_gap
            current_x += group_gap


        plot_data = []
        for model in models:
            for prompt in prompts:
                if (model, prompt) in grouped_pct.index:
                    plot_data.append(grouped_pct.loc[(model, prompt)])
                else:
                    plot_data.append(pd.Series({v: 0 for v in ["true", "false", "missing"]}))
        plot_df = pd.DataFrame(plot_data, index=labels)


        bottom = np.zeros(len(plot_df))
        for flag in ["true", "false", "missing"]:
            ax.bar(x_positions, plot_df[flag], bottom=bottom, width=bar_width, label=flag, color=colors[flag])
            bottom += plot_df[flag].values


        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Percentage of 'Used Internal Knowledge' (%)")
        if len(factcheck) != 0:
            ax.set_title(f"Internal Knowledge Usage (n = {n_items}): {factcheck}, {mode}")
        else:
            ax.set_title(f"Internal Knowledge Usage (n = {n_items}): {mode}")


        
        ax.set_ylim(0, 100)
        ax.legend(title="Used Internal Knowledge")
        plt.tight_layout()
        if len(factcheck) != 0:
            plt.savefig(f"{file_dir}figures/internal_used_{factcheck}_{mode}_{save_suffix}.png", dpi=300)
        else:
            plt.savefig(f"{file_dir}figures/internal_used_{mode}_{save_suffix}.png", dpi=300)
        plt.show()



def plot_confidence_per_mode(file_dir, file_list, save_suffix, factcheck=""):
    records = []
    for file in file_list:
        file_path = file_dir + file
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                model = item["model"]
                mode = item["mode"]
                for prompt in ["basic", "internal", "claim"]:
                    score = item.get(f"response_{prompt}_confidence", None)
                    # ensure numeric
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = np.nan
                    records.append({
                        "model": model,
                        "mode": mode,
                        "prompt": prompt,
                        "confidence": score
                    })

    df = pd.DataFrame(records)

    models = ["llama", "mistral", "qwen"]
    prompts = ["basic", "internal", "claim"]


    colors = {"basic": "#80b1d3", "internal": "#fb8072", "claim": "#b3de69"}


    for mode in sorted(df['mode'].unique()):
        df_mode = df[df['mode'] == mode]

        n_items = int(len(df_mode) / 9)  

        # Compute mean and std per (model, prompt)
        stats = df_mode.groupby(['model', 'prompt'])['confidence'].agg(['mean', 'std']).reset_index()

        fig, ax = plt.subplots(figsize=(5, 5))

        group_gap = 0.4
        bar_width = 0.2
        inside_gap = 0.05

        x_positions = []
        labels = []
        current_x = 0
        for model in models:
            for prompt in prompts:
                x_positions.append(current_x)
                labels.append(f"{model}, {prompt}")
                current_x += bar_width + inside_gap
            current_x += group_gap

        # Align stats with plotting order
        means, stds = [], []
        for model in models:
            for prompt in prompts:
                row = stats[(stats['model'] == model) & (stats['prompt'] == prompt)]
                if not row.empty:
                    means.append(row['mean'].values[0])
                    stds.append(row['std'].values[0])
                else:
                    means.append(0)
                    stds.append(0)

        # Plot bars with error bars
        for i, prompt in enumerate(prompts):
            # Determine which bars correspond to this prompt
            indices = [j for j, label in enumerate(labels) if f", {prompt}" in label]
            ax.bar(
                [x_positions[j] for j in indices],
                [means[j] for j in indices],
                yerr=[stds[j] for j in indices],
                width=bar_width,
                color=colors[prompt],
                label=prompt,
                capsize=4,
                edgecolor='black'
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Confidence Score (mean ± SD)")
        ax.set_ylim(0, 100)
        if len(factcheck) != 0:
            ax.set_title(f"Confidence Scores (n = {n_items}): {factcheck}, {mode}")
        else:
            ax.set_title(f"Confidence Scores (n = {n_items}): {mode}")
        ax.legend(title="Prompt Type")
        plt.tight_layout()
        if len(factcheck) != 0:
            plt.savefig(f"{file_dir}figures/confidence_{factcheck}_{mode}_{save_suffix}.png", dpi=300)
        else:
            plt.savefig(f"{file_dir}figures/confidence_{mode}_{save_suffix}.png", dpi=300)
        plt.show()


def plot_bullet_count_per_mode(file_dir, file_list, save_suffix,  modes = ["2support", "2refute", "mix"], factcheck=""):
    records = []
    mean_dict = {}
    std_dict = {}

    for file in file_list:
        file_path = os.path.join(file_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)

        for item in data:
            model = item["model"]
            mode = item["mode"]
        
            for prompt in ["basic", "internal", "claim"]:
                bullets = item.get(f"response_{prompt}_bullets", [])
        
                if isinstance(bullets, str):
                    bullets = [b.strip() for b in bullets.split("--") if b.strip()]
        
                bullet_count = len(bullets)
        
                records.append({
                    "model": model,
                    "mode": mode,
                    "prompt": prompt,
                    "bullet_count": bullet_count
                })

                

    df = pd.DataFrame(records)
    #print(df.head(5))
    models = ["llama", "mistral", "qwen"]
    prompts = ["basic", "internal", "claim"]
    colors = {"basic": "#80b1d3", "internal": "#fb8072", "claim": "#b3de69"}


    for mode in sorted(df["mode"].unique()):
        df_mode = df[df["mode"] == mode]

        n_items = int(len(df_mode) / 9)  
        stats = df_mode.groupby(["model", "prompt"])["bullet_count"].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(5, 5))

        group_gap = 0.4
        bar_width = 0.2
        inside_gap = 0.05

        x_positions = []
        labels = []
        current_x = 0
        for model in models:
            for prompt in prompts:
                x_positions.append(current_x)
                labels.append(f"{model}, {prompt}")
                current_x += bar_width + inside_gap
            current_x += group_gap


        means, stds = [], []
        for model in models:
            for prompt in prompts:
                row = stats[(stats["model"] == model) & (stats["prompt"] == prompt)]
                if not row.empty:
                    mean = row["mean"].values[0]
                    std = row["std"].values[0]
                    mean_dict[(mode, model, prompt)]= mean
                    std_dict[(mode, model, prompt)]= std
                    means.append(mean)
                    stds.append(std)
                    #print(f"mode {mode}, model {model}, prompt {prompt}: mean (std) is {mean:.3f} ({std:.3f})")
                else:
                    means.append(0)
                    stds.append(0)
                    mean_dict[(mode, model, prompt)]= 0
                    std_dict[(mode, model, prompt)]= 0
                 
            
            
        for i, prompt in enumerate(prompts):
            indices = [j for j, label in enumerate(labels) if f", {prompt}" in label]
            ax.bar(
                [x_positions[j] for j in indices],
                [means[j] for j in indices],
                yerr=[stds[j] for j in indices],
                width=bar_width,
                color=colors[prompt],
                label=prompt,
                capsize=4,
                edgecolor="black"
            )


        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Average Number of Bullets (mean ± SD)")
        if len(factcheck) != 0:           
            ax.set_title(f"Bullet Count (n = {n_items}): {factcheck}, {mode}")
        else:
            ax.set_title(f"Bullet Count (n = {n_items}): {mode}")
        #print(means)
        #ax.set_ylim(0, max(means) * 1.4 if means else 10)
        ax.set_ylim(0, 9)
        ax.legend(title="Prompt Type")

        plt.tight_layout()
        if len(factcheck) != 0: 
            plt.savefig(f"{file_dir}figures/bullet_count_{factcheck}_{mode}_{save_suffix}.png", dpi=300)
        else:
            plt.savefig(f"{file_dir}figures/bullet_count_{mode}_{save_suffix}.png", dpi=300)
        plt.show()

        

    print(modes)
    print('# bullet')
    for model in models:
        line = f"{model}" 
        for mode in modes:
            for prompt in prompts:
                mean = mean_dict[(mode, model, prompt)]
                std = std_dict[(mode, model, prompt)]
                line += f"& {mean:.1f}({std:.1f}) "
        line += '\\''\\'
        print(line)



def plot_bullet_length_per_mode(file_dir, file_list, save_suffix,  modes = ["2support", "2refute", "mix"], factcheck=""):
    """Plot average length of each bullet (in words) per model and prompt."""
    records = []
    mean_dict = {}
    std_dict = {}

    for file in file_list:
        file_path = os.path.join(file_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)

        for item in data:
            model = item["model"]
            mode = item["mode"]

            for prompt in ["basic", "internal", "claim"]:
                #bullets = item.get(f"response_{prompt}_clean", [])
                bullets = item.get(f"response_{prompt}_bullets", [])
                if isinstance(bullets, str):
                    bullets = [b.strip() for b in re.split(r"\s*--\s*", bullets) if b.strip()]

                lengths = [len(b.split()) for b in bullets if b.strip()]
                avg_length = np.mean(lengths) if lengths else 0
                records.append({
                    "model": model,
                    "mode": mode,
                    "prompt": prompt,
                    "avg_bullet_length": avg_length
                })

    df = pd.DataFrame(records)
    models = ["llama", "mistral", "qwen"]
    prompts = ["basic", "internal", "claim"]
    colors = {"basic": "#80b1d3", "internal": "#fb8072", "claim": "#b3de69"}


    for mode in sorted(df["mode"].unique()):
        df_mode = df[df["mode"] == mode]
        n_items = int(len(df_mode) / 9)

        # Mean ± SD of average bullet length per (model, prompt)
        stats = df_mode.groupby(["model", "prompt"])["avg_bullet_length"].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(5, 5))

        group_gap = 0.4
        bar_width = 0.2
        inside_gap = 0.05

        x_positions = []
        labels = []
        current_x = 0
        for model in models:
            for prompt in prompts:
                x_positions.append(current_x)
                labels.append(f"{model}, {prompt}")
                current_x += bar_width + inside_gap
            current_x += group_gap

        means, stds = [], []
        for model in models:
            for prompt in prompts:
                row = stats[(stats["model"] == model) & (stats["prompt"] == prompt)]
                if not row.empty:
                    mean = row["mean"].values[0]
                    std = row["std"].values[0]
                    
                    mean_dict[(mode, model, prompt)]= mean
                    std_dict[(mode, model, prompt)]= std
                    means.append(mean)
                    stds.append(std)
                    
                else:
                    means.append(0)
                    stds.append(0)
                    mean_dict[(mode, model, prompt)]= 0
                    std_dict[(mode, model, prompt)]= 0

        for i, prompt in enumerate(prompts):
            indices = [j for j, label in enumerate(labels) if f", {prompt}" in label]
            ax.bar(
                [x_positions[j] for j in indices],
                [means[j] for j in indices],
                yerr=[stds[j] for j in indices],
                width=bar_width,
                color=colors[prompt],
                label=prompt,
                capsize=4,
                edgecolor="black"
            )


        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Average Bullet Length (words, mean ± SD)")
        ax.set_ylim(0, 25)

        if len(factcheck) != 0:          
            ax.set_title(f"Bullet Length (n = {n_items}): {factcheck}, {mode}")
        else:
            ax.set_title(f"Bullet Length (n = {n_items}): {mode}")
        #ax.set_ylim(0, max(means) * 1.4 if means else 10)
        ax.legend(title="Prompt Type")

        plt.tight_layout()
        if len(factcheck) != 0:  
            plt.savefig(f"{file_dir}figures/bullet_length_{factcheck}_{mode}_{save_suffix}.png", dpi=300)
        else:
            plt.savefig(f"{file_dir}figures/bullet_length_{mode}_{save_suffix}.png", dpi=300)
        plt.show()
    
    print(modes)
    print('\#words per bullet')
    for model in models:
        line = f"{model}" 
        for mode in modes:
            for prompt in prompts:
                mean = mean_dict[(mode, model, prompt)]
                std = std_dict[(mode, model, prompt)]
                line += f"& {mean:.1f}({std:.1f}) "
        line += '\\''\\'
        print(line)

      


def compare_verdicts_between_folders(folder_a, folder_b, suffix_a, suffix_b):

    all_records = []
    models = ["llama", "mistral", "qwen"]

    modes = ["2support", "2refute", "mix"]  


    for model in models:
        for mode in modes:
            #print(f"model is {model}, and mode is {mode}:")
            filename_a = f"{model}_{mode}" + suffix_a
            path_a = os.path.join(folder_a, filename_a)
            filename_b = f"{model}_{mode}" + suffix_b
            path_b = os.path.join(folder_b, filename_b)
    
            if not os.path.exists(path_b):
                print(f"⚠️ No inverse file found for: {path_b}")
                continue
            if not os.path.exists(path_a):
                print(f"⚠️ No inverse file found for: {path_a}")
                continue
    
            with open(path_a) as f:
                data_a = json.load(f)
            with open(path_b) as f:
                data_b = json.load(f)
    

            if len(data_a) != len(data_b):
                print(f"Length mismatch in {filename_a}, skipping.")
                continue
    

            for item_a, item_b in zip(data_a, data_b):
                for prompt in ["basic", "internal", "claim"]:
                    key = f"response_{prompt}_verdict"
    
                    verdict_a = item_a.get(key)
                    verdict_b = item_b.get(key)
    
                    match = (verdict_a == verdict_b)

                    all_records.append({
                        "model": model,
                        "mode": mode,
                        "prompt": prompt,
                        "match": 1.0 if match else 0.0
                    })
    
    return pd.DataFrame(all_records)



def plot_verdict_match_rate(folder_a, folder_b, suffix_a, suffix_b, title_note, save_suffix):

    models = ["llama", "mistral", "qwen"]
    modes = ["2support", "2refute", "mix"]  
    
    prompts = ["basic", "internal", "claim"]
    colors = {"basic": "#80b1d3", "internal": "#fb8072", "claim": "#b3de69"}

    df = compare_verdicts_between_folders(folder_a, folder_b, suffix_a, suffix_b)

    
    for mode in modes:
        df_mode = df[df["mode"] == mode]

        stats = df_mode.groupby(["model", "prompt"])["match"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(5, 5))

        bar_width = 0.23
        spacing = 0.15
        inside_gap = 0.04

        x_positions = []
        labels = []
        cur_x = 0


        for model in models:
            for prompt in prompts:
                x_positions.append(cur_x)
                labels.append(f"{model}-{prompt}")
                cur_x += bar_width + inside_gap
            cur_x += spacing

        values = []
        for model in models:
            for prompt in prompts:
                row = stats[(stats["model"] == model) & (stats["prompt"] == prompt)]
                val = row["match"].values[0] * 100 if not row.empty else 0
                values.append(val)


        for prompt in prompts:
            idxs = [i for i, lbl in enumerate(labels) if f"-{prompt}" in lbl]
            ax.bar(
                [x_positions[i] for i in idxs],
                [values[i] for i in idxs],
                color=colors[prompt],
                width=bar_width,
                label=prompt,
                edgecolor="black",
                linewidth=1
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Verdict Match Rate (%)")

        title = f"{mode}, verdict stability"
        ax.set_title(title)

        ax.set_ylim(0, 100)
        ax.legend()

        plt.tight_layout()

        out_path = os.path.join(folder_a, f"figures/verdict_stability_({title_note})_{mode}_{save_suffix}.png")
        plt.savefig(out_path, dpi=300)
        plt.show()


def plot_inverse_bertscore(file_dir, parsed_list, save_suffix):
    """
    Plot mean ± SD of inverse-pair BERTScore
    using fields: response_<prompt>_inv_bert
    """
    records = []


    for file in parsed_list:
        file_path = os.path.join(file_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)

        for item in data:
            model = item["model"]
            mode = item["mode"]
            _key = f"bert_inv_scores"
            scores = item.get(_key, None)
            if scores is None:
                continue
            for prompt in ["basic", "internal", "claim"]:
                records.append({
                    "model": model,
                    "mode": mode,
                    "prompt": prompt,
                    "inv_bert": float(scores.get(prompt, np.nan))
                })

    df = pd.DataFrame(records)
    models = ["llama", "mistral", "qwen"]
    prompts = ["basic", "internal", "claim"]
    colors = {"basic": "#80b1d3", "internal": "#fb8072", "claim": "#b3de69"}


    for mode in sorted(df["mode"].unique()):
        df_mode = df[df["mode"] == mode]
        n_items = int(len(df_mode) / 9)


        stats = df_mode.groupby(["model", "prompt"])["inv_bert"].agg(["mean", "std"]).reset_index()


        fig, ax = plt.subplots(figsize=(5, 5))

        group_gap = 0.4
        bar_width = 0.2
        inside_gap = 0.05

        x_positions = []
        labels = []
        current_x = 0

        for model in models:
            for prompt in prompts:
                x_positions.append(current_x)
                labels.append(f"{model}, {prompt}")
                current_x += bar_width + inside_gap
            current_x += group_gap


        means, stds = [], []
        for model in models:
            for prompt in prompts:
                row = stats[(stats["model"] == model) & (stats["prompt"] == prompt)]
                if not row.empty:
                    means.append(row["mean"].values[0])
                    stds.append(row["std"].values[0])
                else:
                    means.append(0)
                    stds.append(0)


        for prompt in prompts:
            indices = [i for i, lbl in enumerate(labels) if f", {prompt}" in lbl]
            ax.bar(
                [x_positions[j] for j in indices],
                [means[j] for j in indices],
                yerr=[stds[j] for j in indices],
                width=bar_width,
                color=colors[prompt],
                label=prompt,
                capsize=4,
                edgecolor="black"
            )



        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Inverse-Pair BERTScore (mean ± SD)")
        ax.set_ylim(0, 1)

        title = f"{mode}: Inverse BERT Score (n = {n_items})"
        ax.set_title(title)
        ax.legend(title="Prompt Type")

        plt.tight_layout()


        out_name = f"{file_dir}figures/inverse_bert_{mode}_{save_suffix}.png"
        plt.savefig(out_name, dpi=300)
        plt.show()






def plot_fk_per_mode(file_dir, file_list, save_suffix, 
                    modes = ["2support", "2refute", "mix"],
                      factcheck=""):
    records = []
    mean_dict = {}
    std_dict = {}

    for file in file_list:
        file_path = os.path.join(file_dir, file)
        print("Loading:", file_path)

        with open(file_path, 'r') as f:
            data = json.load(f)

        for item in data:
            model = item.get("model")
            mode = item.get("mode")

            # for each prompt type
            for prompt in ["basic", "internal", "claim"]:
                key = f"response_{prompt}_bullets_fk"   
                if key not in item:
                    continue

                fk_list = item[key]   

                grades = []
                for entry in fk_list:
                    try:
                        g = float(entry.get("fk_grade", np.nan))
                        grades.append(g)
                    except:
                        pass

                avg_fk = np.nanmean(grades) if len(grades) else np.nan

                records.append({
                    "model": model,
                    "mode": mode,
                    "prompt": prompt,
                    "fk_grade": avg_fk
                })

    df = pd.DataFrame(records)
    #print(df.head())

    models = ["llama", "mistral", "qwen"]
    prompts = ["basic", "internal", "claim"]


    colors = {
        "basic":   "#80b1d3",
        "internal":"#fb8072",
        "claim":   "#b3de69"
    }

    for mode in sorted(df["mode"].unique()):
        df_mode = df[df["mode"] == mode]


        n_items = int(len(df_mode) / 9)


        stats = df_mode.groupby(["model", "prompt"])["fk_grade"].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(5, 5))

        group_gap = 0.4
        bar_width = 0.2
        inside_gap = 0.05

        x_positions = []
        labels = []
        current_x = 0


        for model in models:
            for prompt in prompts:
                x_positions.append(current_x)
                labels.append(f"{model}, {prompt}")
                current_x += bar_width + inside_gap
            current_x += group_gap 


        means, stds = [], []
        for model in models:
            for prompt in prompts:
                row = stats[(stats["model"] == model) & (stats["prompt"] == prompt)]
                # if len(row):
                #     means.append(row["mean"].values[0])
                #     stds.append(row["std"].values[0])
                # else:
                #     means.append(0.0)
                #     stds.append(0.0)

                if not row.empty:
                    mean = row["mean"].values[0]
                    std = row["std"].values[0]
                    
                    mean_dict[(mode, model, prompt)]= mean
                    std_dict[(mode, model, prompt)]= std
                    means.append(mean)
                    stds.append(std)
                    
                else:
                    means.append(0)
                    stds.append(0)
                    mean_dict[(mode, model, prompt)]= 0
                    std_dict[(mode, model, prompt)]= 0


        for p_idx, prompt in enumerate(prompts):
            indices = [i for i, lab in enumerate(labels) if f", {prompt}" in lab]

            ax.bar(
                [x_positions[j] for j in indices],
                [means[j] for j in indices],
                yerr=[stds[j] for j in indices],
                width=bar_width,
                color=colors[prompt],
                edgecolor="black",
                capsize=3,
                label=prompt
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Avg Flesch–Kincaid Grade (mean ± SD)")
        ax.set_ylim(bottom=0, top=15)

        if factcheck:
            ax.set_title(f"{factcheck}, {mode}: FK Grade Levels (n={n_items})")
            save_name = f"fk_{factcheck}_{mode}_{save_suffix}.png"
        else:
            ax.set_title(f"{mode}: FK Grade Levels (n={n_items})")
            save_name = f"fk_{mode}_{save_suffix}.png"

        ax.legend(title="Prompt Type")
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, "figures/"+save_name), dpi=300)
        plt.show()

    print(modes)
    print('FK grade')
    for model in models:
        line = model
        for mode in modes:
            for prompt in prompts:
                mean = mean_dict[(mode, model, prompt)]
                std = std_dict[(mode, model, prompt)]
                line += f"& {mean:.1f}({std:.1f}) "
        line += '\\''\\'
        print(line)




def plot_bullet_score_by_evaluator(
    file_dir,
    file_list,
    evaluator_model,
    criterion,
    save_suffix,
    modes = ["2support", "2refute", "mix"],
    plot = True,
):
    records = []
    mean_dict = {}
    std_dict = {}
    invalid_scores_cnt = 0


    for file in file_list:
        file_path = os.path.join(file_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            generator_model = item["model"]
            mode = item["mode"]

            eval_scores = item.get("eval_bullet_scores", {})

            for prompt in ["basic", "internal", "claim"]:
                prompt_scores = eval_scores.get(prompt)
                if not prompt_scores:
                    print(f"{prompt}, generator model {item['model']}, mode {item['mode']}, no prompt_scores")
                    continue


                evaluator_scores = prompt_scores.get(evaluator_model)
                if not evaluator_scores:
                    #print(f"{prompt}, generator model {item["model"]}, mode {item["mode"]}, no scores")
                    print(f"{prompt}, generator model {item['model']}, mode {item['mode']}, no evaluator_scores")
                    continue


                c = evaluator_scores.get(criterion)
                if c is None:
                    #print(f"{prompt}, generator model {item["model"]}, mode {item["mode"]}, no scores")
                    print(f"{prompt}, generator model {item['model']}, mode {item['mode']}, no scores")
                    continue
                #print(c)
                if (c != 1 and c != 0) and (criterion != "relevant_bullets"):
                    print("INVALID SCORE:", {
                        "generator": generator_model,
                        "evaluator": evaluator_model,
                        "mode": mode,
                        "prompt": prompt,
                        "criterion": criterion,
                        "value": c
                    })
                    invalid_scores_cnt += 1
                    continue


                records.append({
                    "generator_model": generator_model,
                    "mode": mode,
                    "prompt": prompt,
                    criterion: c
                })

    df = pd.DataFrame(records)

    generators = ["llama", "mistral", "qwen"]
    prompts = ["basic", "internal", "claim"]
    colors = {"basic": "#80b1d3", "internal": "#fb8072", "claim": "#b3de69"}


    for mode in sorted(df["mode"].unique()):
        df_mode = df[df["mode"] == mode]

        n_items = df_mode.shape[0] // 9  

        stats = (
            df_mode
            .groupby(["generator_model", "prompt"])[criterion]
            .agg(["mean", "std"])
            .reset_index()
        )
        if plot:
            fig, ax = plt.subplots(figsize=(5, 5))

            group_gap = 0.4
            bar_width = 0.2
            inside_gap = 0.05

            x_positions = []
            labels = []
            current_x = 0

            for gen in generators:
                for prompt in prompts:
                    x_positions.append(current_x)
                    labels.append(f"{gen}, {prompt}")
                    current_x += bar_width + inside_gap
                current_x += group_gap

        means, stds = [], []
        for gen in generators:
            for prompt in prompts:
                row = stats[
                    (stats["generator_model"] == gen) &
                    (stats["prompt"] == prompt)
                ]

                if not row.empty:
                    mean = row["mean"].values[0]
                    std = row["std"].values[0]
                    
                    mean_dict[(mode, gen, prompt)]= mean
                    std_dict[(mode, gen, prompt)]= std
                    means.append(mean)
                    stds.append(std)
                    
                else:
                    means.append(0)
                    stds.append(0)
                    mean_dict[(mode, gen, prompt)]= 0
                    std_dict[(mode, gen, prompt)]= 0
        if plot:
            for prompt in prompts:
                idxs = [i for i, l in enumerate(labels) if f", {prompt}" in l]
                ax.bar(
                    [x_positions[i] for i in idxs],
                    [means[i] for i in idxs],
                    yerr=[stds[i] for i in idxs],
                    width=bar_width,
                    color=colors[prompt],
                    capsize=4,
                    edgecolor="black",
                    label=prompt
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel(f"{criterion}, Score (mean ± SD)")
            if criterion == "relevant_bullets":
                ax.set_ylim(0,8)
            else:
                ax.set_ylim(0, 1)

            title = f"{evaluator_model} evaluator | {mode} (n = {n_items}) | {criterion}"
            ax.set_title(title)

            #ax.legend(title="Prompt Type")
            plt.tight_layout()

            outname = f"{file_dir}figures/evaluator({evaluator_model})_{criterion}_{mode}_{save_suffix}.png"
            plt.savefig(outname, dpi=300)
            plt.show()
    

    for gen in generators:
        line = f" ({gen} | {evaluator_model})" 
        for mode in modes:
            for prompt in prompts:
                mean = mean_dict[(mode, gen, prompt)]
                std = std_dict[(mode, gen, prompt)]
                line += f"& {mean:.2f}({std:.2f}) "
        line += '\\''\\'
        print(line)
    print('\hline')

    print("Invalid scores encountered:", invalid_scores_cnt)











def plot_diff_bert_between_ev1_and_ev2(
                    file_dir, 
                    file_list, 
                    save_suffix, 
                    modes = ["2support", "2refute", "mix"],
                    factcheck=""):
    records = []
    mean_dict = {}
    std_dict = {}

    for file in file_list:
        file_path = os.path.join(file_dir, file)
        print("Loading:", file_path)

        with open(file_path, 'r') as f:
            data = json.load(f)

        for item in data:
            model = item.get("model")
            mode = item.get("mode")
            bert_scores = item.get("bert_scores", {})
            # for each prompt type
            for prompt in ["basic", "internal", "claim"]:
                if prompt == "claim":
                    diff_bert_ = 0
                else:
                    s = bert_scores[prompt]
                    diff_bert_ = abs(s.get("evidence_1", None) - s.get("evidence_2", None))

                records.append({
                    "model": model,
                    "mode": mode,
                    "prompt": prompt,
                    "diff_bert": diff_bert_
                })

    df = pd.DataFrame(records)
    #print(df.head())

    models = ["llama", "mistral", "qwen"]
    prompts = ["basic", "internal", "claim"]

    colors = {
        "basic":   "#80b1d3",
        "internal":"#fb8072",
        "claim":   "#b3de69"
    }


    for mode in sorted(df["mode"].unique()):
        df_mode = df[df["mode"] == mode]

        n_items = int(len(df_mode) / 9)


        stats = df_mode.groupby(["model", "prompt"])["diff_bert"].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(5, 5))

        group_gap = 0.4
        bar_width = 0.2
        inside_gap = 0.05

        x_positions = []
        labels = []
        current_x = 0

        for model in models:
            for prompt in prompts:
                x_positions.append(current_x)
                labels.append(f"{model}, {prompt}")
                current_x += bar_width + inside_gap
            current_x += group_gap 

        means, stds = [], []
        for model in models:
            for prompt in prompts:
                row = stats[(stats["model"] == model) & (stats["prompt"] == prompt)]

                if not row.empty:
                    mean = row["mean"].values[0]
                    std = row["std"].values[0]
                    
                    mean_dict[(mode, model, prompt)]= mean
                    std_dict[(mode, model, prompt)]= std
                    means.append(mean)
                    stds.append(std)
                    
                else:
                    means.append(0)
                    stds.append(0)
                    mean_dict[(mode, model, prompt)]= 0
                    std_dict[(mode, model, prompt)]= 0


        for p_idx, prompt in enumerate(prompts):
            indices = [i for i, lab in enumerate(labels) if f", {prompt}" in lab]

            ax.bar(
                [x_positions[j] for j in indices],
                [means[j] for j in indices],
                yerr=[stds[j] for j in indices],
                width=bar_width,
                color=colors[prompt],
                edgecolor="black",
                capsize=3,
                label=prompt
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Difference of Bert scores (mean ± SD)")
        ax.set_ylim(bottom=0, top=0.3)

        if factcheck:
            ax.set_title(f"{factcheck}, {mode}: Difference of Bert scores (n={n_items})")
            save_name = f"bert_diff_{factcheck}_{mode}_{save_suffix}.png"
        else:
            ax.set_title(f"{mode}: Difference of Bert scores (n={n_items})")
            save_name = f"bert_diff_{mode}_{save_suffix}.png"

        ax.legend(title="Prompt Type")
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, "figures/"+save_name), dpi=300)
        plt.show()

    print(modes)
    print('Difference of Bert scores')
    for model in models:
        line = model
        for mode in modes:
            for prompt in prompts:
                mean = mean_dict[(mode, model, prompt)]
                std = std_dict[(mode, model, prompt)]
                line += f"& {mean:.2f}({std:.2f}) "
        line += '\\''\\'
        print(line)

