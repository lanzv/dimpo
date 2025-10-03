#!/bin/bash

base_models=(
    "llama3_1b_instruct"
    "llama3_3b_instruct"
    "llama3_8b_instruct"
    "qwen3_4b_instruct"
    "qwen2_7b_instruct"
)


projection_types=(
    "PCA"
    "Rand"
    "SimPO"
    "ORPO"
    "CPO"
    "DimPO"
)

for model_name in "${base_models[@]}"; do
    if [[ "$model_name" == "llama3_1b_instruct" ]]; then
        target_dim_list=(32 16 8 4 2 1)
    else
        target_dim_list=(64 32 16 8 4 2 1)
    fi

    for projection_type in "${projection_types[@]}"; do
        for target_dim in "${target_dim_list[@]}"; do
            python3 run_projection_experiment.py \
                --model_name "$model_name" \
                --target_dim "$target_dim" \
                --projection_type "$projection_type" \
                --experimental_mode "projection_performance"
        done
    done
done