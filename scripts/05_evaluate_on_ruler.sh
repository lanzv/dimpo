#!/bin/bash

models=("llama3_1b_instruct" "llama3_3b_instruct" "llama3_8b_instruct" "qwen3_4b_instruct")


# Total number of attention layers (N)
declare -A total_attn_layers
total_attn_layers["llama3_1b_instruct"]=16
total_attn_layers["llama3_3b_instruct"]=28
total_attn_layers["llama3_8b_instruct"]=32
total_attn_layers["qwen3_4b_instruct"]=36

# Target dimensions (d')
declare -A target_dims_map
target_dims_map["llama3_1b_instruct"]="32"
target_dims_map["llama3_3b_instruct"]="64"
target_dims_map["llama3_8b_instruct"]="64"
target_dims_map["qwen3_4b_instruct"]="64"

# Number of attention layers (l) to reduce its dimensionality for each model
declare -A num_layers_to_reduce_map
num_layers_to_reduce_map["llama3_1b_instruct"]="0 4 6 8"
num_layers_to_reduce_map["llama3_3b_instruct"]="0 7 12 14"
num_layers_to_reduce_map["llama3_8b_instruct"]="0 8 12 16"
num_layers_to_reduce_map["qwen3_4b_instruct"]="0 9 15 18"





# Ruler tasks and their maximum sequence lengths
declare -A ruler_tasks
ruler_tasks["ruler4096"]=4096
ruler_tasks["ruler8192"]=8192


for model_name in "${models[@]}"; do
    total_layers=${total_attn_layers[$model_name]}
    echo "Processing model: ${model_name} with ${total_layers} attention layers."

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
                *)
                    echo "  Skipping unknown model: ${model_name}"
                    continue
                    ;;
            esac

            for ruler_task in "${!ruler_tasks[@]}"; do
                max_seq_len=${ruler_tasks[$ruler_task]}
                monitoring_dir="./results/long_context/ruler/monitoring/${ruler_task}"
                output_path="./results/long_context/ruler/scores/${ruler_task}/${model_name}/${d_prime}_${l}"
                
                python_cmd="python3 ./evaluate_harness.py \
                    --model LowDimAttentionsModel \
                    --model_args \"max_length=32768,attn_implementation=eager,parallelize=True,pretrained=${model_path},monitoring_dir=${monitoring_dir},dtype=bfloat16,trust_remote_code=True,lowdim_attentions_path=${checkpoint_path},lowdim_attn_layers=${lowdim_attn_layers_list}\" \
                    --tasks ruler \
                    --output_path ${output_path} \
                    --metadata='{\"max_seq_lengths\":[${max_seq_len}]}' \
                    --device cuda --batch_size 1 --apply_chat_template \
                    --trust_remote_code"
                
                echo "  Executing command for ${ruler_task}: ${python_cmd}"
                eval "${python_cmd}"
            done
        done
    done
done





# RULER for longer contexts, to compare with MagicPIG
#    applicable only for l=0 (original model) due to 
#    computational limitation of attention implementation 
#    - eager is not effective enough for such long contexts
#    - flash_attention_2 and sdpa are not meant to be for key and value vectors of different shape


# Define the models and their respective configurations
declare -A total_attn_layers
total_attn_layers["llama3_1b_instruct"]=16
total_attn_layers["llama3_3b_instruct"]=28
total_attn_layers["llama3_8b_instruct"]=32

# Target dimensions (d')
declare -A target_dims_map
target_dims_map["llama3_1b_instruct"]="32"
target_dims_map["llama3_3b_instruct"]="64"
target_dims_map["llama3_8b_instruct"]="64"

# Number of attention layers (l) to reduce, specified as only 0 (original models)
declare -A num_layers_to_reduce_map
num_layers_to_reduce_map["llama3_1b_instruct"]="0"
num_layers_to_reduce_map["llama3_3b_instruct"]="0"
num_layers_to_reduce_map["llama3_8b_instruct"]="0"

# Ruler tasks with extended sequence lengths
declare -A ruler_tasks
ruler_tasks["ruler16384"]=16384
ruler_tasks["ruler32768"]=32768
ruler_tasks["ruler65536"]=65536


for model_name in "llama3_1b_instruct" "llama3_3b_instruct" "llama3_8b_instruct"; do
    total_layers=${total_attn_layers[$model_name]}
    echo "Processing model: ${model_name} with ${total_layers} attention layers."

    read -ra model_dims <<< "${target_dims_map[$model_name]}"
    read -ra model_layers_to_reduce <<< "${num_layers_to_reduce_map[$model_name]}"

    for d_prime in "${model_dims[@]}"; do
        for l in "${model_layers_to_reduce[@]}"; do
            echo "  Running evaluation for d'=${d_prime}, l=${l} layers..."

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
                *)
                    echo "  Skipping unknown model: ${model_name}"
                    continue
                    ;;
            esac

            # Loop through the new ruler tasks
            for ruler_task in "${!ruler_tasks[@]}"; do
                max_seq_len=${ruler_tasks[$ruler_task]}
                monitoring_dir="./results/long_context/ruler/monitoring/${ruler_task}"
                output_path="./results/long_context/ruler/scores/${ruler_task}/${model_name}/${d_prime}_${l}"
                
                # !! flash_attention_2 !! does not support differnt dimensions of values and keys
                # sdpa supports it but requires much more memory for the attention computation than when shapes match
                # -> use eager for fair comparision and if you have enough of GPU RAM memory, only if l==0 use flash_attention_2 or sdpa
                python_cmd="python3 ./evaluate_harness.py \
                    --model LowDimAttentionsModel \
                    --model_args \"max_length=70000,attn_implementation=flash_attention_2,parallelize=True,pretrained=${model_path},monitoring_dir=${monitoring_dir},dtype=bfloat16,trust_remote_code=True,lowdim_attentions_path=${checkpoint_path},lowdim_attn_layers=${lowdim_attn_layers_list}\" \
                    --tasks ruler \
                    --output_path ${output_path} \
                    --metadata='{\"max_seq_lengths\":[${max_seq_len}]}' \
                    --device cuda --batch_size 1 --apply_chat_template \
                    --trust_remote_code"
                
                echo "  Executing command for ${ruler_task}: ${python_cmd}"
                eval "${python_cmd}"
            done
        done
    done
done