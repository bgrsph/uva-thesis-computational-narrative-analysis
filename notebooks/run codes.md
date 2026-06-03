  # --- setup (run once, on the cluster, from repo root) ---
  cd /home/bsipahioglu/uva-thesis-computational-narrative-analysis
  export EXPERIMENT_DIR=/home/bsipahioglu/uva-thesis-computational-narrative-analysis/data/experiments/experiment_test_9385_20260602_2124
  export SPLIT=test
  mkdir -p $EXPERIMENT_DIR/logs        # SLURM won't create the log dir itself

  # --- causal ---
  sbatch --partition=gpu_a100 -J llama-causal-n50 \
    -o $EXPERIMENT_DIR/logs/%x-%j.out -e $EXPERIMENT_DIR/logs/%x-%j.err \
    models/llama/infer_relations.sbatch \
    --input       $EXPERIMENT_DIR/tma_subset_events_processed.${SPLIT}.jsonl \
    --output      $EXPERIMENT_DIR/llama_runs/causal.jsonl \
    --condition   causal \
    --sample-size 50

  # --- temporal ---
  sbatch --partition=gpu_a100 -J llama-temporal-n50 \
    -o $EXPERIMENT_DIR/logs/%x-%j.out -e $EXPERIMENT_DIR/logs/%x-%j.err \
    models/llama/infer_relations.sbatch \
    --input       $EXPERIMENT_DIR/tma_subset_events_processed.${SPLIT}.jsonl \
    --output      $EXPERIMENT_DIR/llama_runs/temporal.jsonl \
    --condition   temporal \
    --sample-size 50

  # --- temporal_causal_joint ---
  sbatch --partition=gpu_a100 -J llama-tcjoint-n50 \
    -o $EXPERIMENT_DIR/logs/%x-%j.out -e $EXPERIMENT_DIR/logs/%x-%j.err \
    models/llama/infer_relations.sbatch \
    --input       $EXPERIMENT_DIR/tma_subset_events_processed.${SPLIT}.jsonl \
    --output      $EXPERIMENT_DIR/llama_runs/temporal_causal_joint.jsonl \
    --condition   temporal_causal_joint \
    --sample-size 50


  Commands (relative EXPERIMENT_DIR, from repo root, --sample-size 300):
cd /home/bsipahioglu/uva-thesis-computational-narrative-analysis
export EXPERIMENT_DIR=data/experiments/experiment_test_9385_20260602_2124
export SPLIT=test
echo "[$EXPERIMENT_DIR]"   # must be data/... (relative), not /Users or /home

for C in causal temporal temporal_causal_joint; do
  sbatch --partition=gpu_a100 -J llama-$C-n300 \
    models/llama/infer_relations.sbatch \
    --input       $EXPERIMENT_DIR/tma_subset_events_processed.${SPLIT}.jsonl \
    --output      $EXPERIMENT_DIR/llama_runs/$C.jsonl \
    --condition   $C \
    --sample-size 300
done


For 20 minutes of expected time: 


  sbatch --partition="$SLURM_PARTITION_GPU" --time=00:60:00 -J embed-e5_mistral
  models/embed/encode_embeddings.sbatch --encoder e5_mistral --input $EXPERIMENT_DIR/experiment.jsonl --output $EXPERIMENT_DIR/embeddings/e5_mistral.jsonl


  sbatch --partition="$SLURM_PARTITION_GPU" --time=00:60:00 -J embed-sbert_mpnet
  models/embed/encode_embeddings.sbatch --encoder sbert_mpnet --input $EXPERIMENT_DIR/experiment.jsonl --output $EXPERIMENT_DIR/embeddings/sbert_mpnet.jsonl


  sbatch --partition="$SLURM_PARTITION_GPU" --time=00:60:00 -J embed-story_emb
  models/embed/encode_embeddings.sbatch --encoder story_emb --input $EXPERIMENT_DIR/experiment.jsonl --output $EXPERIMENT_DIR/embeddings/story_emb.jsonl


  sbatch --partition="$SLURM_PARTITION_GPU" --time=00:60:00 -J embed-qwen3_emb_0p6b
  models/embed/encode_embeddings.sbatch --encoder qwen3_emb_0p6b --input $EXPERIMENT_DIR/experiment.jsonl --output $EXPERIMENT_DIR/embeddings/qwen3_emb_0p6b.jsonl

