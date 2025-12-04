#!/bin/bash

DATA="data/sudoku-extreme-1k-aug-1000"
EPOCHS=50000
EVAL_INTERVAL=5000
LR=1e-4
WD=1.0
BATCH_SIZE=512

# Experiment 1: Baseline
echo "==================== Starting Exp 1: Baseline ===================="
python pretrain.py \
  arch=trm \
  data_paths="[${DATA}]" \
  evaluators="[]" \
  epochs=${EPOCHS} eval_interval=${EVAL_INTERVAL} \
  lr=${LR} puzzle_emb_lr=${LR} \
  weight_decay=${WD} puzzle_emb_weight_decay=${WD} \
  global_batch_size=${BATCH_SIZE} \
  +run_name="exp1_baseline_trm" \
  ema=True

# Experiment 2: Qwen3 Gate Only
echo "==================== Starting Exp 2: Qwen3 Gate ===================="
python pretrain.py \
  arch=trm_gated_qwen \
  data_paths="[${DATA}]" \
  evaluators="[]" \
  epochs=${EPOCHS} eval_interval=${EVAL_INTERVAL} \
  lr=${LR} puzzle_emb_lr=${LR} \
  weight_decay=${WD} puzzle_emb_weight_decay=${WD} \
  global_batch_size=${BATCH_SIZE} \
  +run_name="exp2_qwen3_gate" \
  ema=True

# Experiment 3: Recurrence Gate Only
echo "==================== Starting Exp 3: Recurrence Gate ===================="
python pretrain.py \
  arch=trm_gated_recurrence \
  data_paths="[${DATA}]" \
  evaluators="[]" \
  epochs=${EPOCHS} eval_interval=${EVAL_INTERVAL} \
  lr=${LR} puzzle_emb_lr=${LR} \
  weight_decay=${WD} puzzle_emb_weight_decay=${WD} \
  global_batch_size=${BATCH_SIZE} \
  +run_name="exp3_recurrence_gate" \
  ema=True

# Experiment 4: Hybrid
echo "==================== Starting Exp 4: Hybrid Gates ===================="
python pretrain.py \
  arch=trm_hybrid_gated \
  data_paths="[${DATA}]" \
  evaluators="[]" \
  epochs=${EPOCHS} eval_interval=${EVAL_INTERVAL} \
  lr=${LR} puzzle_emb_lr=${LR} \
  weight_decay=${WD} puzzle_emb_weight_decay=${WD} \
  global_batch_size=${BATCH_SIZE} \
  +run_name="exp4_hybrid_gated" \
  ema=True

echo "==================== All experiments completed ===================="