#!/bin/bash


models=("llama3_1b_instruct" "llama3_3b_instruct" "llama3_8b_instruct" "qwen3_4b_instruct" "qwen2_7b_instruct")

# Total number of attention layers (N)
declare -A total_attn_layers
total_attn_layers["llama3_1b_instruct"]=16
total_attn_layers["llama3_3b_instruct"]=28
total_attn_layers["llama3_8b_instruct"]=32
total_attn_layers["qwen3_4b_instruct"]=36
total_attn_layers["qwen2_7b_instruct"]=28

# Target dimensions (d')
declare -A target_dims_map
target_dims_map["llama3_1b_instruct"]="32 16 4 1"
target_dims_map["llama3_3b_instruct"]="64 16 4 1"
target_dims_map["llama3_8b_instruct"]="64 16 4 1"
target_dims_map["qwen3_4b_instruct"]="64 16 4 1"
target_dims_map["qwen2_7b_instruct"]="64 16 4 1"

# Number of attention layers (l) to reduce its dimensionality for each model
declare -A num_layers_to_reduce_map
num_layers_to_reduce_map["llama3_1b_instruct"]="0 2 4 6 8 10 12 14 16"
num_layers_to_reduce_map["llama3_3b_instruct"]="0 2 4 7 10 12 14 16 18 21 24 26 28"
num_layers_to_reduce_map["llama3_8b_instruct"]="0 2 4 8 12 16 20 24 28 30 32"
num_layers_to_reduce_map["qwen3_4b_instruct"]="0 2 4 6 9 12 15 18 21 24 27 30 32 34 36"
num_layers_to_reduce_map["qwen2_7b_instruct"]="0 2 4 7 10 12 14 16 18 21 24 26 28"




for model_name in "${models[@]}"; do
    total_layers=${total_attn_layers[$model_name]}
    read -ra model_dims <<< "${target_dims_map[$model_name]}"
    read -ra model_layers_to_reduce <<< "${num_layers_to_reduce_map[$model_name]}"

    for d_prime in "${model_dims[@]}"; do
        for l in "${model_layers_to_reduce[@]}"; do
            echo "  Running evaluation for d'=${d_prime}, l=${l} layers..."

            # Construct the lowdim_attn_layers argument
            lowdim_attn_layers_list=""
            if [ "$l" -gt 0 ]; then
                start_layer=$((total_layers - l))
                for ((i = start_layer; i < total_layers; i++)); do
                    if [[ -n "$lowdim_attn_layers_list" ]]; then
                        lowdim_attn_layers_list+=" "
                    fi
                    lowdim_attn_layers_list+="${i}"
                done
            fi

            # Set model path and checkpoint path based on model name
            model_path=""
            checkpoint_path=""
            case "${model_name}" in
                "llama3_1b_instruct")
                    model_path="meta-llama/Llama-3.2-1B-Instruct"
                    checkpoint_path="./results/checkpoints/DimPO_llama3_1b_instruct_${d_prime}"
                    ;;
                "llama3_3b_instruct")
                    model_path="meta-llama/Llama-3.2-3B-Instruct"
                    checkpoint_path="./results/checkpoints/DimPO_llama3_3b_instruct_${d_prime}"
                    ;;
                "llama3_8b_instruct")
                    model_path="meta-llama/Llama-3.1-8B-Instruct"
                    checkpoint_path="./results/checkpoints/DimPO_llama3_8b_instruct_${d_prime}"
                    ;;
                "qwen3_4b_instruct")
                    model_path="Qwen/Qwen3-4B-Instruct-2507"
                    checkpoint_path="./results/checkpoints/DimPO_qwen3_4b_instruct_${d_prime}"
                    ;;
                "qwen2_7b_instruct")
                    model_path="Qwen/Qwen2.5-7B-Instruct"
                    checkpoint_path="./results/checkpoints/DimPO_qwen2_7b_instruct_${d_prime}"
                    ;;
                *)
                    echo "  Skipping unknown model: ${model_name}"
                    continue
                    ;;
            esac

            python_cmd="python3 ./evaluate_harness.py \
                --model LowDimAttentionsModel \
                --model_args \"pretrained=${model_path},monitoring_dir=./results/general_tasks/monitoring2,trust_remote_code=True,dtype=bfloat16,lowdim_attentions_path=${checkpoint_path},lowdim_attn_layers=${lowdim_attn_layers_list}\" \
                --tasks hellaswag,arc_challenge,mmlu,truthfulqa_mc2,winogrande \
                --output_path ./results/general_tasks/scores/${model_name}/${d_prime}_${l} \
                --num_fewshot 0 --device auto --batch_size 32 --trust_remote_code"
            
            echo "  Executing command: ${python_cmd}"
            #eval "${python_cmd}"
        done
    done
done
