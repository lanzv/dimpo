from src.models.lowdim_modules import (
    LowDimPCAFactory,
    LowDimDimPOFactory, 
    LowDimRandFactory,
    LowDimSimPOFactory, 
    LowDimTripletFactory, 
    LowDimCPOFactory,
    LowDimORPOFactory,
    LowDimKLFactory
)
from src.models.dimpo_kl import LowDimDimPOKLFactory
from src.models.lowdim_trainer import LowDimTrainer
import logging
import torch
import os
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import MethodType

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)




# register new models here
MODEL_PATHS = {
    "llama3_1b_instruct": "../models/Llama-3.2-1B-Instruct",
    "llama3_3b_instruct": "../models/Llama-3.2-3B-Instruct",
    "llama3_8b_instruct": "../models/Llama-3.1-8B-Instruct",
    "qwen3_4b_instruct": "../models/Qwen3-4B-Instruct-2507",
    "qwen2_7b_instruct": "../models/Qwen2.5-7B-Instruct",
}

# default set to auto if not specified by this dict or by parameter args.device_map
DEVICE_MAP = {
    "llama3_1b_instruct": "cuda:0"
}

OUTPUT_FOLDERS = {
    "projection_performance": "./results/dimension_reduction/projection_performance",
    "selection_all_key_pairs": "./results/dimension_reduction/key_pair_selection/all_key_pairs",
    "selection_multiple_distinct_pairs": "./results/dimension_reduction/key_pair_selection/multiple_distinct_pairs",
    "selection_level_of_key_diversity": "./results/dimension_reduction/key_pair_selection/level_of_key_diversity",
}






def run_experiment(lowdim_module_factory, model_path, 
                    scores_file_name, output_dir="./output_scores",
                    epochs=1, train_instances=10, device_map="auto",
                    num_train_tokens_per_instance=4096, eval_on_devset=False,
                    checkpoint_path=None, evaluate=True):
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map)
    
    trainer = LowDimTrainer(
        model=model, 
        lowdim_module_factory=lowdim_module_factory, 
        tokenizer=tokenizer,
        num_train_instances=train_instances, num_train_tokens_per_instance=num_train_tokens_per_instance,
        num_val_instances=1, num_val_tokens_per_instance=32,
        epochs=epochs,
        train_batch_size=1, eval_batch_size=1,
        lowdim_attn_layers = None, # [24, 25, 26, 27] for replacing attention layers with ids 24, 25, 26 and 27; None for replacing all layers
    )
    
    trainer.train()

    if checkpoint_path is not None:
        os.makedirs(checkpoint_path, exist_ok=True)
        trainer.save_lowdim_checkpoints(checkpoint_path)

    if evaluate:
        scores = trainer.eval(num_instances=10, num_tokens_per_instance=4096, batch_size=1, split="validation" if eval_on_devset else "test")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, scores_file_name)
        with open(output_path, 'w') as f:
            json.dump(scores, f)




def main(args):
    model_path = MODEL_PATHS[args.model_name]
    target_dim = args.target_dim
    projection_type = args.projection_type
    if args.device_map == None:
        device_map = DEVICE_MAP.get(args.model_name, "auto")
    else:
        device_map = args.device_map
    
    checkpoint_path = args.checkpoint_path

    experimental_mode = args.experimental_mode
    num_sampled_keys = args.num_sampled_keys
    key_vector_pair_distance = args.key_vector_pair_distance
    num_sampled_pairs = args.num_sampled_pairs
    output_dir = OUTPUT_FOLDERS[experimental_mode]
    eval_on_devset = False if experimental_mode == "projection_performance" else True
    num_train_tokens_per_instance = 4096 if experimental_mode == "projection_performance" else 128
    alias = ""
    if experimental_mode == "selection_all_key_pairs":
        alias += f"_{num_sampled_keys}"
    if experimental_mode == "selection_multiple_distinct_pairs":
        alias += f"_{num_sampled_pairs}"
    if experimental_mode == "selection_level_of_key_diversity":
        alias += f"_{key_vector_pair_distance}"


    if projection_type == "PCA": # The same number of training instances as the rest of projection types
        logging.info("=========================================")
        logging.info(f"\t\tPCA {target_dim}, Full Queries: False")
        logging.info("=========================================\n\n\n\n\n")
        lowdim_module_factory = LowDimPCAFactory(target_dim=target_dim, full_queries=False)
        scores_file_name = f"PCA_{args.model_name}_{target_dim}{alias}.json"

    elif projection_type == "Rand":
        logging.info("=========================================")
        logging.info(f"\t\tRand {target_dim}")
        logging.info("=========================================\n\n\n\n\n")
        lowdim_module_factory = LowDimRandFactory(target_dim=target_dim)
        scores_file_name = f"Rand_{args.model_name}_{target_dim}{alias}.json" 

    elif projection_type == "DimPO":
        # L_DimPO = L_list + lambda * L_head. k=0, lambda=0 is the listwise term alone.
        k = args.k
        lmbda = 0.0 if k == 0 else args.lmbda
        logging.info("=========================================")
        logging.info(f"\t\tDimPO {target_dim}, k={k}, lambda={lmbda}")
        logging.info("=========================================\n\n\n\n\n")
        lowdim_module_factory = LowDimDimPOFactory(
                            target_dim=target_dim, beta=1.0, gamma=0.0001, lr=0.0001, batch_size=1,
                            k=k, lmbda=lmbda, num_sampled_keys=num_sampled_keys
                        )
        if k == 0:
            scores_file_name=f"DimPO_k0_{args.model_name}_{target_dim}{alias}.json"
        elif k == 64 and lmbda == 1.0:
            scores_file_name=f"DimPO_{args.model_name}_{target_dim}{alias}.json"
        else:
            scores_file_name=f"DimPO_k{k}_w{lmbda}_{args.model_name}_{target_dim}{alias}.json"

    elif projection_type == "SimPO":
        logging.info("=========================================")
        logging.info(f"\t\tSimPO {target_dim}")
        logging.info("=========================================\n\n\n\n\n")
        lowdim_module_factory = LowDimSimPOFactory(
                            target_dim=target_dim, beta=1.0, gamma=1.0, lr=0.001, batch_size=32,
                            experimental_mode=experimental_mode, num_sampled_keys=num_sampled_keys,
                            key_vector_pair_distance=key_vector_pair_distance,num_sampled_pairs=num_sampled_pairs
                        ) 
        scores_file_name = f"SimPO_{args.model_name}_{target_dim}{alias}.json"

    elif projection_type == "Triplet":
        logging.info("=========================================")
        logging.info(f"\t\tTriplet {target_dim}")
        logging.info("=========================================\n\n\n\n\n")
        lowdim_module_factory = LowDimTripletFactory(
                            target_dim=target_dim, margin=0.1, lr=0.0001, batch_size=32,
                            experimental_mode=experimental_mode, num_sampled_keys=num_sampled_keys,
                            key_vector_pair_distance=key_vector_pair_distance, num_sampled_pairs=num_sampled_pairs
                        )
        scores_file_name = f"Triplet_{args.model_name}_{target_dim}{alias}.json" 

    elif projection_type == "ORPO":
        logging.info("=========================================")
        logging.info(f"\t\tORPO {target_dim}")
        logging.info("=========================================\n\n\n\n\n")
        lowdim_module_factory = LowDimORPOFactory(
                            target_dim=target_dim, lmbda=0.1, lr=0.001, batch_size=32,
                            experimental_mode=experimental_mode, num_sampled_keys=num_sampled_keys,
                            key_vector_pair_distance=key_vector_pair_distance, num_sampled_pairs=num_sampled_pairs
                        ) 
        scores_file_name = f"ORPO_{args.model_name}_{target_dim}{alias}.json"

    elif projection_type == "CPO":
        logging.info("=========================================")
        logging.info(f"\t\tCPO {target_dim}")
        logging.info("=========================================\n\n\n\n\n")
        lowdim_module_factory = LowDimCPOFactory(
                            target_dim=target_dim, beta=1.0, lmbda=0.1, lr=0.0001, batch_size=32,
                            experimental_mode=experimental_mode, num_sampled_keys=num_sampled_keys,
                            key_vector_pair_distance=key_vector_pair_distance, num_sampled_pairs=num_sampled_pairs
                        )
        scores_file_name = f"CPO_{args.model_name}_{target_dim}{alias}.json"
    elif projection_type == "KL":
        logging.info("=========================================")
        logging.info(f"\t\tKL {target_dim}")
        logging.info("=========================================\n\n\n\n\n")
        lowdim_module_factory = LowDimKLFactory(
                            target_dim=target_dim, lr=0.0001, batch_size=1,
                            num_sampled_keys=num_sampled_keys
                        )
        scores_file_name = f"KL_{args.model_name}_{target_dim}{alias}.json"

    elif projection_type == "DimPO_KL":
        logging.info("=========================================")
        logging.info(f"\t\tDimPO_KL {target_dim}, kl_weight={args.kl_weight}")
        logging.info("=========================================\n\n\n\n\n")
        lowdim_module_factory = LowDimDimPOKLFactory(
                            target_dim=target_dim, beta=1.0, gamma=0.0001, lr=0.0001, batch_size=1,
                            kl_weight=args.kl_weight, num_sampled_keys=num_sampled_keys
                        )
        scores_file_name = f"DimPO_KL_{args.model_name}_{target_dim}{alias}.json"

    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")


    run_experiment(lowdim_module_factory=lowdim_module_factory, scores_file_name=scores_file_name,
                    model_path=model_path, epochs=1, train_instances=10, device_map=device_map,
                    num_train_tokens_per_instance=num_train_tokens_per_instance, output_dir=output_dir, eval_on_devset=eval_on_devset,
                    checkpoint_path=checkpoint_path, evaluate=args.evaluate)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("--model_name", required=True, help="Name of the model.")
    parser.add_argument("--target_dim", type=int, required=True, help="Target dimension for the experiment.")
    parser.add_argument("--projection_type", required=True, help="PCA, Rand, DimPO, DimPO_KL, KL, SimPO, ORPO, CPO, Triplet.")
    parser.add_argument("--k", type=int, default=64, required=False, help="Head size k in L_head. Default 64. Use 0 for the listwise term only (lambda is then 0).")
    parser.add_argument("--lmbda", type=float, default=1.0, required=False, help="lambda on L_head. Default 1. Ignored when --k 0.")
    parser.add_argument("--kl_weight", type=float, default=1.0, required=False, help="Weight of the full KL term for the DimPO_KL ablation.")
    parser.add_argument("--experimental_mode", type=str, default="projection_performance", required=False, help="[projection_performance, selection_all_key_pairs, selection_multiple_distinct_pairs, selection_level_of_key_diversity]")
    parser.add_argument("--checkpoint_path", type=str, default=None, required=False, help="If checkpoint path is None, the model is evaluated but not saved. If checkpoint path is specified, the model is not evaluted, but saved.")
    parser.add_argument("--disable_evaluation", dest="evaluate", action="store_false", help="Set this flag to skip evaluation mode (default is to evaluate).")
    parser.add_argument("--num_sampled_keys", type=int, default=None, required=False, help="applicable only when selection_all_key_pairs is set as an experimental mode")
    parser.add_argument("--num_sampled_pairs", type=int, default=None, required=False, help="applicable only when selection_multiple_distinct_pairs is set as an experimental mode")
    parser.add_argument("--key_vector_pair_distance", type=int, default=None, required=False, help="applicable only when selection_level_of_key_diversity is set as an experimental mode")
    parser.add_argument("--device_map", type=str, default=None, required=False, help="[auto, cuda:0, cuda:1, .., cuda]")
    args = parser.parse_args()
    main(args)
