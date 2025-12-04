run_name="exp2_qwen3_gate"
python pretrain.py \
arch=trm_gated_qwen \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 \
weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
global_batch_size=512 \
+run_name=${run_name} \
ema=True
# run_name="pretrain_att_sudoku"
# python pretrain.py \
# arch=trm \
# data_paths="[data/sudoku-extreme-1k-aug-1000]" \
# evaluators="[]" \
# epochs=50000 eval_interval=5000 \
# lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
# arch.L_layers=2 \
# global_batch_size=512 \
# arch.H_cycles=3 arch.L_cycles=6 \
# +run_name=${run_name} ema=True