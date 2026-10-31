#!/usr/bin/env bash
# Sweep the ResMLP nonlinear-branch CL coefficient; linear-branch CL stays at 0.5.
# Starts training immediately; uses seeds 2021 through 2025.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO/datasets/time_series}"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/genTS:$REPO/genTS/external/tslib:${PYTHONPATH:-}"

lr=5e-4
dropout_rate=0.0
n_train_step=5000
lookback_win=720
model_name=mlp_nobias

SCRIPT=$REPO/consistency_loss/run_forecasting_res_nl.py

for seed in 2021 2022 2023 2024 2025; do
for regu_coef_mlp in 0 0.5 2 8 32 128; do
for future_win in 96 192 336 720; do
    echo "Running seed=$seed regu_coef_mlp=$regu_coef_mlp pred_len=$future_win"


    python -u "$SCRIPT" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$DATA_ROOT/ETT-small/" \
        --data_path ETTh1.csv \
        --model_id ETTh1 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 2048 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --depth 3 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --seed $seed \
        --val_type branch_res \
        --individual False \
        --decouple_checker decouple \
        --regu_coef 0.5 \
        --regu_coef_mlp $regu_coef_mlp \
        --dropout $dropout_rate

    python -u "$SCRIPT" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$DATA_ROOT/ETT-small/" \
        --data_path ETTh2.csv \
        --model_id ETTh2 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 2048 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --depth 3 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --seed $seed \
        --val_type branch_res \
        --individual False \
        --decouple_checker decouple \
        --regu_coef 0.5 \
        --regu_coef_mlp $regu_coef_mlp \
        --dropout $dropout_rate

    python -u "$SCRIPT" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$DATA_ROOT/ETT-small/" \
        --data_path ETTm1.csv \
        --model_id ETTm1 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 2048 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --depth 3 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --seed $seed \
        --val_type branch_res \
        --individual False \
        --decouple_checker decouple \
        --regu_coef 0.5 \
        --regu_coef_mlp $regu_coef_mlp \
        --dropout $dropout_rate

    python -u "$SCRIPT" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$DATA_ROOT/ETT-small/" \
        --data_path ETTm2.csv \
        --model_id ETTm2 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 2048 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --depth 3 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef 0.5 \
        --seed $seed \
        --val_type branch_res \
        --individual False \
        --decouple_checker decouple \
        --regu_coef_mlp $regu_coef_mlp \
        --dropout $dropout_rate

    python -u "$SCRIPT" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$DATA_ROOT/electricity/" \
        --data_path electricity.csv \
        --model_id ECL \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 2048 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --learning_rate $lr \
        --depth 3 \
        --train_steps $n_train_step \
        --seed $seed \
        --val_type branch_res \
        --individual False \
        --decouple_checker decouple \
        --regu_coef 0.5 \
        --regu_coef_mlp $regu_coef_mlp \
        --dropout $dropout_rate

    python -u "$SCRIPT" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$DATA_ROOT/traffic/" \
        --data_path traffic.csv \
        --model_id traffic \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 2048 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --learning_rate $lr \
        --depth 3 \
        --train_steps $n_train_step \
        --seed $seed \
        --val_type branch_res \
        --individual False \
        --decouple_checker decouple \
        --regu_coef 0.5 \
        --regu_coef_mlp $regu_coef_mlp \
        --dropout $dropout_rate

    python -u "$SCRIPT" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$DATA_ROOT/weather/" \
        --data_path weather.csv \
        --model_id weather \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $future_win \
        --batch_size 128 \
        --d_model 2048 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --depth 3 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --regu_coef 0.5 \
        --seed $seed \
        --val_type branch_res \
        --individual True \
        --decouple_checker decouple \
        --regu_coef_mlp $regu_coef_mlp \
        --dropout $dropout_rate



done
done
done
