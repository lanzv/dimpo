#!/bin/bash

base_models=(
    "llama3_1b_instruct"
    "llama3_3b_instruct"
    "llama3_8b_instruct"
    "qwen3_4b_instruct"
    "qwen2_7b_instruct"
)

for model_name in "${base_models[@]}"; do
    if [[ "$model_name" == "llama3_1b_instruct" ]]; then
        target_dim_list=(32 16 8 4 2 1)
    else
        target_dim_list=(64 32 16 8 4 2 1)
    fi

    for target_dim in "${target_dim_list[@]}"; do
        python3 run_projection_experiment.py \
            --model_name "$model_name" \
            --target_dim "$target_dim" \
            --projection_type "DimPO" \
            --checkpoint_path "./results/checkpoints/DimPO_${model_name}_${target_dim}" \
            --disable_evaluation
    done
done
