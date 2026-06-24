# Quantifying the Impact of Temporal and Causal Structures on Narrative Similarity Retrieval

A Hybrid Event-Based Framework Combining Supervised Event Extraction and LLM-Inferred Temporo-Causal Relations for Structured Narrative Representation

MSc thesis, Information Studies (Data Science), University of Amsterdam.

- Author: Buğra Sipahioğlu (14318334)
- Supervisor: Alexandra Barancová (University of Amsterdam)

## Overview

This repository contains the full pipeline for the thesis. It extracts events from narrative summaries with a supervised BERT+CRF model, infers temporal and causal relations between those events with Llama-3-8B, linearizes the result into text, embeds it, and evaluates narrative-similarity retrieval on the Tell Me Again! dataset. The evaluation compares event-based and relation-enriched representations against lexical and dense baselines, on both a non-pseudonymized and a pseudonymized version of the corpus.

## Pipeline steps

1. Event detection (MAVEN): train a BERT+CRF event extractor on the MAVEN corpus.
2. Dataset subsetting: build the eligible Tell Me Again! retrieval corpus.
3. Event extraction: run the trained BERT+CRF over each summary, then drop summaries with fewer than five events and re-enforce at least two summaries per work.
4. Relation annotation: Llama-3-8B zero-shot temporal, causal, and joint temporo-causal relations, followed by complete-case and hallucinated-relation filtering.
5. Linearization: convert events and relations into one text string per condition, then drop summaries whose linearized form exceeds the embedder window.
6. Embedding: encode every condition with E5-Mistral, Qwen3-0.6B, and StoryEmb.
7. Baselines: BoW and TF-IDF over the full text and over the event triggers.
8. Evaluation: retrieval metrics (P@1, Hits@10, R-Precision, MAP, NDCG), computed in a separate notebook.

## Environments

The project uses three Python environments. The pipeline notebook documents the exact setup commands for each, so follow it for step-by-step details.

- Pipeline: `venv/` at the repo root, Python 3.14. Runs the notebooks and every local step. Install with `pip install -r requirements.txt`.
- BERT+CRF: `models/bert_crf/.venv-maven-train`, Python 3.9, pinned to transformers 4.18. Used only for BERT+CRF training and event extraction. Install with `pip install -r models/bert_crf/requirements.txt`.
- Cluster: `.venv/` on the Snellius HPC cluster, used only for the cluster option of the Llama and embedding steps.

## How to run

The notebooks are numbered in run order: `1.pipeline.ipynb`, then `2.evaluation.ipynb`, then the diagnostics. In detail:

1. `notebooks/1.pipeline.ipynb`, run once per dataset arm, top to bottom:
   - Non-pseudonymized arm: run Section 2a only, then continue through the rest of the notebook.
   - Pseudonymized arm: run Section 2a first, then Section 2b (Section 2b builds the pseudonymized counterpart of the Section 2a run), then continue through the rest of the notebook.

   Each pass writes one `experiment.jsonl` under `data/experiments/<experiment_name>/`.
2. `notebooks/2.evaluation.ipynb`, run once after both pipeline passes. It reads both experiment folders and produces the matched, side-by-side retrieval results.
3. Diagnostics, run after the steps above and independent of each other (either order):
   - `notebooks/3.error_analysis.ipynb` reads the evaluation outputs, so run it after `2.evaluation.ipynb`.
   - `notebooks/manual_annotation.ipynb` needs only the event-extraction output from `1.pipeline.ipynb`.

## Notebooks

- `notebooks/1.pipeline.ipynb`: the main pipeline (steps 1 to 7 above), run once per arm.
- `notebooks/2.evaluation.ipynb`: retrieval evaluation; reads both arms and produces the matched results tables. Run once, after both pipeline passes.
- `notebooks/3.error_analysis.ipynb`: diagnostic analyses of the retrieval errors and of the inferred relation structure. Run after `2.evaluation.ipynb`.
- `notebooks/manual_annotation.ipynb`: manual evaluation of the BERT+CRF event extraction on a sample of summaries (trigger identification and event-type classification). Needs only the event output from `1.pipeline.ipynb`.
- `notebooks/eda.ipynb`: exploratory data analysis of MAVEN and Tell Me Again! (corpus statistics, token counts, event-count thresholds) that motivates the subsetting choices. Standalone.

## Data and models

- Tell Me Again! dataset: https://github.com/uhh-lt/tell-me-again (installed via `pip install tell_me_again`).
- BERT+CRF event extractor, based on the MAVEN baseline: https://github.com/THU-KEG/MAVEN-dataset/tree/main/baselines/BERT+CRF (base model https://huggingface.co/bert-base-uncased). The code in `models/bert_crf/` is the upstream baseline with a minimal edits to run on a modern arm64 (`transformers==4.18.0`, Python 3.9). Each change is annotated with a `# PATCHED:` comment (`grep -rn PATCHED models/bert_crf/`), and no training logic was changed.
- Llama-3-8B-Instruct, relation inference: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct (gated; requires access approval and a Hugging Face login).
- E5-Mistral-7B-Instruct, encoder: https://huggingface.co/intfloat/e5-mistral-7b-instruct
- Qwen3-Embedding-0.6B, encoder: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- StoryEmb, narrative-specific encoder baseline: https://huggingface.co/uhhlt/story-emb (paper: https://aclanthology.org/2024.emnlp-main.339)