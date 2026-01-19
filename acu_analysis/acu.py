import json
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Softmax over {true, false, maybe}
def softmax(logits_dict):
    max_logit = max(logits_dict.values())
    exps = {k: math.exp(v - max_logit) for k, v in logits_dict.items()}
    s = sum(exps.values())
    return {k: v / s for k, v in exps.items()}


# delta P calculation (paper formula)
def delta_p(p_with, p_without):
    out = {}
    for t in p_with.keys():
        pw = p_with[t]       # P(t | C,E)
        po = p_without[t]   # P(t | C)

        if pw >= po:
            out[t] = (pw - po) / (1 - po) if po < 1 else 0.0
        else:
            out[t] = (pw - po) / po if po > 0 else 0.0

    return out




def get_D_map(mode):
    if mode == "2support":
        return {
            "true": 1.0,
            "false": -1.0,
            "maybe": -1.0
        }
    elif mode == "2refute":
        return {
            "false": 1.0,
            "true": -1.0,
            "maybe": -1.0
        }
    else:
        return None   # for "mix"



def compute_acu(delta_p_dict, D_map):
    """
    delta_p_dict: {"true": ΔP, "false": ΔP, "maybe": ΔP}
    D_map:         {"true": ±1, "false": ±1, "maybe": ±1}
    """
    T = list(delta_p_dict.keys())   # should be 3 tokens
    acu = 0.0
    for t in T:
        acu += D_map[t] * delta_p_dict[t]
    acu /= len(T)
    return acu





def draw_single_p(model, mode, _dir, _suffix, save_folder, title_suffix, save_file_suffix):
    
    _file = f"{model}_{mode}" + _suffix
    file_path = _dir + _file
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        items = data
    else:
        items = data["data"]
    
    

    rows = []
    cnt = len(items)
    
    for i, item in enumerate(items):
    
        logits_basic = item["logits_basic"]
        logits_internal = item["logits_internal"]
        logits_claim = item["logits_claim"]
    
        p_basic = softmax(logits_basic)
        p_internal = softmax(logits_internal)
        p_claim = softmax(logits_claim)
    
        #  delta P 
        delta_basic = delta_p(p_basic, p_claim)
        delta_internal = delta_p(p_internal, p_claim)
    
        # long-form rows
        for label in p_claim.keys():
            rows.append({
                "sample_id": i,
                "token": label,
    
                "p_claim": p_claim[label],
                "p_basic": p_basic[label],
                "p_internal": p_internal[label],
    
                "delta_p_basic": delta_basic[label],
                "delta_p_internal": delta_internal[label],
            })
    

    df = pd.DataFrame(rows)

    
    plt.figure(figsize=(4, 3))
    sns.histplot(data=df, x="delta_p_internal", hue="token", bins=40)
        
    plt.xlabel(r"$\Delta P$")
    plt.title(f"{model}, {mode}, n = {cnt}, internal {title_suffix}")
    plt.tight_layout()
    plt.savefig(f"../acu_analysis/figures/{save_folder}/p_dist_internal_{model}_{mode}_{save_file_suffix}.png")
    plt.show()
    
    
    plt.figure(figsize=(4, 3))
    sns.histplot(data=df, x="delta_p_basic", hue="token", bins=40)
    plt.xlabel(r"$\Delta P$")
    plt.title(f"{model}, {mode}, n = {cnt}, basic {title_suffix}")
    plt.tight_layout()
    plt.savefig(f"../acu_analysis/figures/{save_folder}/p_dist_basic_{model}_{mode}_{save_file_suffix}.png",     
                bbox_inches="tight",
                dpi=300
               )
    plt.show()






def gen_long_df(models, mode, _dir, _suffix):
    all_rows = []   #accumulator
    for model in models:   
        _file = f"{model}_{mode}"  + _suffix
        file_path = _dir + _file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        items = data if isinstance(data, list) else data["data"]

        for i, item in enumerate(items):

            logits_basic = item["logits_basic"]
            logits_internal = item["logits_internal"]
            logits_claim = item["logits_claim"]

            # --- softmax ---
            p_basic = softmax(logits_basic)
            p_internal = softmax(logits_internal)
            p_claim = softmax(logits_claim)

            # --- ΔP ---
            delta_basic = delta_p(p_basic, p_claim)
            delta_internal = delta_p(p_internal, p_claim)

            # STORE model + mode + label + ΔP
            for label in p_claim.keys():
                all_rows.append({
                    "sample_id": i,
                    "model": model,
                    "mode": mode,
                    "token": label,
                    "delta_p_basic": delta_basic[label],
                    "delta_p_internal": delta_internal[label],
                })

    df = pd.DataFrame(all_rows)
    
    df_long = pd.melt(
        df,
        id_vars=["sample_id", "model", "mode", "token"],
        value_vars=["delta_p_basic", "delta_p_internal"],
        var_name="delta_type",
        value_name="delta_p"
    )
    
    df_long["delta_type"] = df_long["delta_type"].str.replace("delta_p_", "")
    df_long["model_evidence"] = df_long["model"] + "-" + df_long["delta_type"]
    return df_long



def plot_p_boxplots(df_long, mode, save_folder, save_file_suffix):

    sub = df_long[df_long["mode"] == mode]
    cnt = int(len(sub)/3/6)
    hue_order = [
        "llama-basic", "llama-internal",
        "mistral-basic", "mistral-internal",
        "qwen-basic", "qwen-internal"
    ]
    sns.boxplot(
        data=sub,
        x="token",
        y="delta_p",
        hue="model_evidence",
        hue_order=hue_order, 
        width=0.3,
        flierprops=dict(
            marker='o',
            markersize=3,     # smaller outliers
            linestyle='none',
            markerfacecolor='gray',
            alpha=0.6
        )
    )
    
    plt.title(f"ΔP Distribution, {mode}, n = {cnt}, {save_file_suffix}")
    plt.xlabel("Token")
    plt.ylabel("ΔP")
    plt.legend(
        title="Model + Evidence",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,              
        frameon=True
    )
    plt.tight_layout()
    #if save:
    plt.savefig(f"../acu_analysis/figures/{save_folder}/p_boxplot_{mode}_{save_file_suffix}.png",
                bbox_inches="tight",
                dpi=300)
    plt.show()







    
def draw_single_acu(model, mode, _dir, _suffix, save_folder, title_suffix, save_file_suffix):

    if mode == "mix":
        print(f"[SKIP] ACU undefined for mix: {model}, {save_file_suffix}")
        return

    _file = f"{model}_{mode}" + _suffix

    file_path = _dir + _file

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else data["data"]
    cnt = len(items)

    D_map = get_D_map(mode)

    acu_basic_vals = []
    acu_internal_vals = []

    for item in items:

        p_basic = softmax(item["logits_basic"])
        p_internal = softmax(item["logits_internal"])
        p_claim = softmax(item["logits_claim"])

        delta_basic = delta_p(p_basic, p_claim)
        delta_internal = delta_p(p_internal, p_claim)

        acu_basic = compute_acu(delta_basic, D_map)
        acu_internal = compute_acu(delta_internal, D_map)

        item["acu_basic"] = float(acu_basic)
        item["acu_internal"] = float(acu_internal)

        acu_basic_vals.append(acu_basic)
        acu_internal_vals.append(acu_internal)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"ACU written into: {file_path}")

    plt.figure(figsize=(4, 3))
    sns.histplot(acu_internal_vals, bins=40)
    plt.xlabel("ACU")
    plt.title(f"{model}, {mode}, n = {cnt},  internal {title_suffix}")
    plt.tight_layout()
    #if save:
    plt.savefig(f"../acu_analysis/figures/{save_folder}/acu_internal_{model}_{mode}_{save_file_suffix}.png", dpi=300)
    plt.show()

    plt.figure(figsize=(4, 3))
    sns.histplot(acu_basic_vals, bins=40)
    plt.xlabel("ACU")
    plt.title(f"{model}, {mode}, n = {cnt}, basic {title_suffix}")
    plt.tight_layout()
    plt.savefig(f"../acu_analysis/figures/{save_folder}/acu_basic_{model}_{mode}_{save_file_suffix}.png", 
                bbox_inches="tight",
                dpi=300)
    plt.show()





def draw_acu_summary(acu_type, models, modes, _dir, _suffix, save_folder, save_file_suffix):
    """
    acu_type: "basic" or "internal"
    verdict_mode: "verdict_only" or "verdict_first"
    """

    means = {mode: [] for mode in modes}
    stds  = {mode: [] for mode in modes}
    Ns    = {mode: 0 for mode in modes}

    # LOAD ALL ACU STATS
    for mode in modes:
        if mode == "mix":
            print(f"acu summary--------[SKIP] ACU undefined for mix: {save_file_suffix}")
            break
        for model in models:

                
            _file = f"{model}_{mode}" + _suffix
            file_path = _dir + _file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data["data"]

            vals = [
                item[f"acu_{acu_type}"]
                for item in items
                if item.get(f"acu_{acu_type}") is not None
            ]

            means[mode].append(np.mean(vals))
            stds[mode].append(np.std(vals))
            Ns[mode] = len(vals)


    # support group at x = 0,1,2
    # refute  group at x = 5,6,7  (big gap)
    x_support = np.array([0, 1, 2])
    x_refute  = np.array([5, 6, 7])

    plt.figure(figsize=(4, 3))

    plt.bar(
        x_support,
        means["2support"],
        yerr=stds["2support"],
        capsize=6,
        width=0.6,
        label="2support"
    )

    plt.bar(
        x_refute,
        means["2refute"],
        yerr=stds["2refute"],
        capsize=6,
        width=0.6,
        label="2refute"
    )

    all_x = np.concatenate([x_support, x_refute])
    model_labels = models + models
    plt.xticks(all_x, model_labels, rotation=45)

    ax = plt.gca()                        # current axis
    ax2 = ax.twiny()                     
    ax2.spines["bottom"].set_visible(False)

    # Move second axis to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))  
    
    # Remove extra spines and ticks except bottom
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="x", length=0)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([np.mean(x_support), np.mean(x_refute)])
    ax2.set_xticklabels(["2support", "2refute"], fontsize=11)

    ax.set_ylim(0, 1)

    #plt.xticks(x, labels, rotation=45, ha="right", fontsize=9)
    plt.axhline(0, linestyle="--", linewidth=1)

    ax.set_ylabel("ACU (mean ± std)")

    
    plt.title(f"{acu_type}, {save_file_suffix}")
    plt.tight_layout()
    plt.savefig(f"../acu_analysis/figures/{save_folder}/acu_summary_{acu_type}_{save_file_suffix}.png", 
                bbox_inches="tight",
                dpi=300)
    plt.show()




def draw_acu_summary_combined(models, modes, _dir, _suffix, save_folder, save_file_suffix):
    """
    Draw one figure:
    - x-axis: models grouped by 2support / 2refute
    - bars: basic vs internal ACU side by side
    """

    acu_types = ["basic", "internal"]
    means = {acu: {mode: [] for mode in modes} for acu in acu_types}
    stds  = {acu: {mode: [] for mode in modes} for acu in acu_types}

    for acu in acu_types:
        for mode in modes:
            if mode == "mix":
                continue
            for model in models:
                _file = f"{model}_{mode}" + _suffix
                file_path = _dir + _file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                items = data if isinstance(data, list) else data["data"]
                vals = [
                    item[f"acu_{acu}"]
                    for item in items
                    if item.get(f"acu_{acu}") is not None
                ]

                means[acu][mode].append(np.mean(vals))
                stds[acu][mode].append(np.std(vals))


    n_models = len(models)
    bar_w = 0.35

    x_support = np.arange(n_models)
    x_refute  = np.arange(n_models) + n_models + 1.5  

    offset_basic   = -bar_w / 2
    offset_internal =  bar_w / 2

    plt.figure(figsize=(6, 3.5))
    COLOR_BASIC = "tab:blue"
    COLOR_INTERNAL = "tab:orange"



    # 2support
    plt.bar(
        x_support + offset_basic,
        means["basic"]["2support"],
        yerr=stds["basic"]["2support"],
        width=bar_w,
        capsize=5,
        color=COLOR_BASIC,
        label="basic"
    )
    
    plt.bar(
        x_support + offset_internal,
        means["internal"]["2support"],
        yerr=stds["internal"]["2support"],
        width=bar_w,
        capsize=5,
        color=COLOR_INTERNAL,
        label="internal"
    )

    # 2refute (same colors, no labels)
    plt.bar(
        x_refute + offset_basic,
        means["basic"]["2refute"],
        yerr=stds["basic"]["2refute"],
        width=bar_w,
        capsize=5,
        color=COLOR_BASIC
    )
    
    plt.bar(
        x_refute + offset_internal,
        means["internal"]["2refute"],
        yerr=stds["internal"]["2refute"],
        width=bar_w,
        capsize=5,
        color=COLOR_INTERNAL
    )


    all_x = np.concatenate([x_support, x_refute])
    xtick_labels = models + models
    plt.xticks(all_x, xtick_labels, rotation=45)

    plt.ylabel("ACU (mean ± std)")
    plt.ylim(0, 1)
    plt.axhline(0, linestyle="--", linewidth=1)

    # group labels
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([x_support.mean(), x_refute.mean()])
    ax2.set_xticklabels(["2support", "2refute"])
    ax2.xaxis.set_ticks_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 40))
    ax2.tick_params(axis="x", length=0)
    for spine in ["top", "left", "right"]:
        ax2.spines[spine].set_visible(False)

    #plt.legend(frameon=False, ncol=2)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="C0", label="basic"),
        Patch(facecolor="C1", label="internal")
    ]
    
    plt.legend(handles=legend_elements, frameon=False, ncol=2)

    plt.title("ACU statistics (V_prompt)")

    plt.tight_layout()
    plt.savefig(
        f"../acu_analysis/figures/{save_folder}/acu_summary_combined_{save_file_suffix}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

