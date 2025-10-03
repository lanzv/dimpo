models=("magicpig_llama3_1b_instruct" "magicpig_llama3_3b_instruct" "magicpig_llama3_8b_instruct")

# Total number of attention layers (N)
declare -A total_attn_layers
total_attn_layers["magicpig_llama3_1b_instruct"]=16
total_attn_layers["magicpig_llama3_3b_instruct"]=28
total_attn_layers["magicpig_llama3_8b_instruct"]=32

# Target dimensions (d')
declare -A target_dims_map
target_dims_map["magicpig_llama3_1b_instruct"]="32"
target_dims_map["magicpig_llama3_3b_instruct"]="64"
target_dims_map["magicpig_llama3_8b_instruct"]="64"

# Number of attention layers (l) to reduce its dimensionality for each model
declare -A num_layers_to_reduce_map
num_layers_to_reduce_map["magicpig_llama3_1b_instruct"]="0 6"
num_layers_to_reduce_map["magicpig_llama3_3b_instruct"]="0 12"
num_layers_to_reduce_map["magicpig_llama3_8b_instruct"]="0 12"


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
                "magicpig_llama3_1b_instruct")
                    model_path="meta-llama/Llama-3.2-1B-Instruct"
                    checkpoint_path="../results/checkpoints/DimPO_llama3_1b_instruct_${d_prime}"
                    ;;
                "magicpig_llama3_3b_instruct")
                    model_path="meta-llama/Llama-3.2-3B-Instruct"
                    checkpoint_path="../results/checkpoints/DimPO_llama3_3b_instruct_${d_prime}"
                    ;;
                "magicpig_llama3_8b_instruct")
                    model_path="meta-llama/Llama-3.1-8B-Instruct"
                    checkpoint_path="../results/checkpoints/DimPO_llama3_8b_instruct_${d_prime}"
                    ;;
                *)
                    echo "  Skipping unknown model: ${model_name}"
                    continue
                    ;;
            esac

            monitoring_dir="../results/long_context/longbench/monitoring"
            output_path="../results/long_context/longbench/scores/${model_name}/${d_prime}_${l}"
            
            python_cmd="python3 ./evaluate_harness.py \
                --model magicpig \
                --model_args \"max_length=70000,pretrained=${model_path},monitoring_dir=${monitoring_dir},dtype=bfloat16,trust_remote_code=True,lowdim_attentions_path=${checkpoint_path},lowdim_attn_layers=${lowdim_attn_layers_list}\" \
                --tasks longbench \
                --output_path ${output_path} \
                --device cuda --batch_size 1 --apply_chat_template \
                --trust_remote_code"
            
            echo "  Executing command for longbench: ${python_cmd}"
            eval "${python_cmd}"
        done
    done
done
