---
license: mit
language: en
pretty_name: DRUID
configs:
  - config_name: DRUID
    data_files:
    - split: train
      path: "druid.tsv"
  - config_name: DRUID+
    data_files:
    - split: train
      path: "druid_plus.tsv"
---

# Dataset Card for 🧙🏽‍♀️DRUID (Dataset of Retrieved Unreliable, Insufficient and Difficult-to-understand context)

## Dataset Details 

More details on the dataset can be found in our paper: https://arxiv.org/abs/2412.17031.

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

DRUID contains real-world (query, context) pairs to facilitate studies of context usage and failures in real-world RAG scenarios. The dataset is based on the prototypical task of automated claim verification, for which automated retrieval of real-world evidence is crucial. Therefore, we sometimes also refer to 'query' as 'claim' and 'context' as 'evidence'.

## Uses

- Evaluate model context usage.
- Test automated methods for claim verification.

## Dataset Structure

We release two versions of the dataset: DRUID and DRUID+. DRUID is a high-quality (corresponding to top Cohere reranker scores) subset of DRUID+ manually annotated for evidence relevance and stance. The dataset contains the following columns:

- id: A unique identifier for each dataset sample. It also indicates claim source.
- claim_id: A unique identifier for each claim in the dataset. One claim may correspond to multiple samples, corresponding to different evidence pieces retrieved from different webpages.
- claim_source: The fact-check site article from which the sample claim was retrieved.
- claim: A claim/query that is some statement about the world.
- claimant: The person/organisation behind the claim.
- claim_date: The date at which the claim was posted by the fact-check site (so the claim may have been released some time before that).
- evidence_source: The webpage from which the sample evidence has been retrieved.
- evidence: Evidence/context intended for assessing the veracity of the given claim.
- evidence_data: The publish date of the webpage the evidence has been retrieved from.
- factcheck_verdict: The fact-checked verdict with respect to the claim, does not have to align with the evidence stance.
- is_gold: Whether the evidence has been retrieved from the corresponding fact-check site or "retrieved from the wild".
- relevant: Whether the evidence is relevant with respect to the given claim. Has been manually annotated for the DRUID samples.
- evidence_stance: The stance of the evidence, i.e. whether it supports the claim or not, or is insufficient. Has been manually annotated for the DRUID samples.

Samples based on claims from 'borderlines' follow a sligthly different structure, as the "claims" in this case are based on the [borderlines](https://github.com/manestay/borderlines) dataset. Therefore, they have no corresponding claimant, claim source or fact-check verdict, etc.

## Dataset Creation

### Claim Collection

We sample claims verified by fact-checkers using [Google's Factcheck API](https://developers.google.com/fact-check/tools/api/reference/rest). We only sample claims in English. The claims are collected from 7 diverse fact-checking sources, representing science, politics, Northern Ireland, Sri Lanka, the US, India, France, etc. All claims have been assessed by human fact-checkers. We also collect claims based on the [borderlines](https://github.com/manestay/borderlines) dataset, for which we can expect there to be more frequent conflicts between contexts.

### Evidence Collection

For each claim in DRUID DRUID+, we retrieve up to 5 and 40 snippets of evidence, respectively. First, a gold-standard evidence document is retrieved from the original fact-checking site, which is the 'summary' of the fact-checking article written by the author of the article. For the remaining snippets of evidence, we use an automated retrieval method. We collect the top 20 search results for each of the Google and Bing search engines. The found webpages are then chunked into paragraphs and reranked by the Cohere rerank model (`rerank-english-v3.0' from [here](https://docs.cohere.com/v2/docs/rerank-2)). Evidence corresponding to the top-ranked chunks is included in DRUID.

### Relevance and Stance Annotation

Since the evidence is collected using automated retrieval, as opposed to controlled synthesis, we need to assess the relevance of the retrieved information to the claim, and, if it is relevant, what stance it represents. For this, we crowd-source evidence-level annotations using [Prolific](https://www.prolific.com/) and [Potato](https://github.com/davidjurgens/potato). Each evidence piece in DRUID is double annotated for _relevance_ (_relevant_ or _not relevant_) and _stance_ to the claim (_supports_, _insufficient-supports_, _insufficient-neutral_, _insufficient-contradictory_, _insufficient-refutes_ or _refutes_).

The annotator compensation was approximately 9 GBP/hour (the compensation was fixed for each task while the annotator completion time varied).

## Citation

```
@misc{druid,
      title={A Reality Check on Context Utilisation for Retrieval-Augmented Generation}, 
      author={Lovisa Hagström and Sara Vera Marjanović and Haeun Yu and Arnav Arora and Christina Lioma and Maria Maistro and Pepa Atanasova and Isabelle Augenstein},
      year={2024},
      eprint={2412.17031},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.17031}, 
}
```