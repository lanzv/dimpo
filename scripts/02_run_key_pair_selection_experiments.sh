#!/bin/bash

run_experiment() {
    local model_name="$1"
    local experimental_mode="$2"
    local specific_arg_name="$3"
    local specific_arg_list=("${@:4}")

    if [[ "$experimental_mode" == "selection_all_key_pairs" ]]; then
        local current_projection_types=("SimPO" "Triplet" "ORPO" "CPO" "DimPO")
    else
        local current_projection_types=("SimPO" "Triplet" "ORPO" "CPO")
    fi

    if [[ "$model_name" == "llama3_1b_instruct" ]]; then
        target_dim_list=(32 16 8 4 2 1)
    else
        target_dim_list=(64 32 16 8 4 2 1)
    fi

    for projection_type in "${current_projection_types[@]}"; do
        for target_dim in "${target_dim_list[@]}"; do
            for specific_arg in "${specific_arg_list[@]}"; do
                python3 run_projection_experiment.py \
                    --model_name "$model_name" \
                    --target_dim "$target_dim" \
                    --projection_type "$projection_type" \
                    --experimental_mode "$experimental_mode" \
                    --"$specific_arg_name" "$specific_arg"
            done
        done
    done
}

base_models=("llama3_1b_instruct" "llama3_3b_instruct" "llama3_8b_instruct" "qwen3_4b_instruct" "qwen2_7b_instruct")

# All Key Pairs
num_sam_keys_list=(2 4 8 16)
for model_name in "${base_models[@]}"; do
    run_experiment "$model_name" "selection_all_key_pairs" "num_sampled_keys" "${num_sam_keys_list[@]}"
done

# Multiple Distinct Pairs
num_sam_pairs_list=(1 2 4 8 16 32 64)
for model_name in "${base_models[@]}"; do
    run_experiment "$model_name" "selection_multiple_distinct_pairs" "num_sampled_pairs" "${num_sam_pairs_list[@]}"
done

# Level of Key Diversity
distance_list=(1 8 16 32 64 128)
for model_name in "${base_models[@]}"; do
    run_experiment "$model_name" "selection_level_of_key_diversity" "key_vector_pair_distance" "${distance_list[@]}"
done
