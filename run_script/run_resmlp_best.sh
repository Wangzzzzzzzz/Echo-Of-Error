#!/usr/bin/env bash
# Run ResMLP with the configured dataset/horizon CL coefficients.
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

run_experiment() {
    local root_path=$1
    local data_path=$2
    local model_id=$3
    local data=$4
    local enc_in=$5
    local dec_in=$6
    local c_out=$7
    local pred_len=$8
    local regu_coef_mlp=$9
    local individual=${10}
    local seed=${11}

    echo "Running ${model_id} seed=$seed pred_len=$pred_len regu_coef_mlp=$regu_coef_mlp"
    python -u "$SCRIPT" \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$root_path" \
        --data_path "$data_path" \
        --model_id "$model_id" \
        --model $model_name \
        --data "$data" \
        --features M \
        --seq_len $lookback_win \
        --label_len 0 \
        --pred_len $pred_len \
        --batch_size 128 \
        --d_model 2048 \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --depth 3 \
        --learning_rate $lr \
        --train_steps $n_train_step \
        --seed $seed \
        --val_type branch_res \
        --individual $individual \
        --decouple_checker decouple \
        --regu_coef 0.5 \
        --regu_coef_mlp $regu_coef_mlp \
        --dropout $dropout_rate
}

for seed in 2021 2022 2023 2024 2025; do

    # ==================== ETTh1 ====================
    # Best regu_coef_mlp: 96->32, 192->32, 336->128, 720->128
    run_experiment "$DATA_ROOT/ETT-small/" ETTh1.csv ETTh1 ETTh1 7 7 7  96  32 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTh1.csv ETTh1 ETTh1 7 7 7 192  32 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTh1.csv ETTh1 ETTh1 7 7 7 336 128 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTh1.csv ETTh1 ETTh1 7 7 7 720 128 False $seed

    # ==================== ETTh2 ====================
    # Best regu_coef_mlp: 96->32, 192->128, 336->128, 720->128
    run_experiment "$DATA_ROOT/ETT-small/" ETTh2.csv ETTh2 ETTh2 7 7 7  96  32 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTh2.csv ETTh2 ETTh2 7 7 7 192 128 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTh2.csv ETTh2 ETTh2 7 7 7 336 128 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTh2.csv ETTh2 ETTh2 7 7 7 720 128 False $seed

    # ==================== ETTm1 ====================
    # Best regu_coef_mlp: 96->8, 192->32, 336->32, 720->32
    run_experiment "$DATA_ROOT/ETT-small/" ETTm1.csv ETTm1 ETTm1 7 7 7  96   8 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTm1.csv ETTm1 ETTm1 7 7 7 192  32 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTm1.csv ETTm1 ETTm1 7 7 7 336  32 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTm1.csv ETTm1 ETTm1 7 7 7 720  32 False $seed

    # ==================== ETTm2 ====================
    # Best regu_coef_mlp: 96->8, 192->8, 336->8, 720->8
    run_experiment "$DATA_ROOT/ETT-small/" ETTm2.csv ETTm2 ETTm2 7 7 7  96 8 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTm2.csv ETTm2 ETTm2 7 7 7 192 8 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTm2.csv ETTm2 ETTm2 7 7 7 336 8 False $seed
    run_experiment "$DATA_ROOT/ETT-small/" ETTm2.csv ETTm2 ETTm2 7 7 7 720 8 False $seed

    # ==================== ECL ====================
    # Best regu_coef_mlp: 96->0.5, 192->0.5, 336->0.5, 720->0
    run_experiment "$DATA_ROOT/electricity/" electricity.csv ECL custom 321 321 321  96 0.5 False $seed
    run_experiment "$DATA_ROOT/electricity/" electricity.csv ECL custom 321 321 321 192 0.5 False $seed
    run_experiment "$DATA_ROOT/electricity/" electricity.csv ECL custom 321 321 321 336 0.5 False $seed
    run_experiment "$DATA_ROOT/electricity/" electricity.csv ECL custom 321 321 321 720   0 False $seed

    # ==================== Traffic ====================
    # Best regu_coef_mlp: 96->0, 192->0, 336->0, 720->0
    run_experiment "$DATA_ROOT/traffic/" traffic.csv traffic custom 862 862 862  96 0 False $seed
    run_experiment "$DATA_ROOT/traffic/" traffic.csv traffic custom 862 862 862 192 0 False $seed
    run_experiment "$DATA_ROOT/traffic/" traffic.csv traffic custom 862 862 862 336 0 False $seed
    run_experiment "$DATA_ROOT/traffic/" traffic.csv traffic custom 862 862 862 720 0 False $seed

    # ==================== Weather ====================
    # Best regu_coef_mlp: 96->32, 192->32, 336->32, 720->128
    run_experiment "$DATA_ROOT/weather/" weather.csv weather custom 21 21 21  96  32 True $seed
    run_experiment "$DATA_ROOT/weather/" weather.csv weather custom 21 21 21 192  32 True $seed
    run_experiment "$DATA_ROOT/weather/" weather.csv weather custom 21 21 21 336  32 True $seed
    run_experiment "$DATA_ROOT/weather/" weather.csv weather custom 21 21 21 720 128 True $seed

done
