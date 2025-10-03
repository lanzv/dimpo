from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from torch.utils.data import DataLoader
import torch
import logging
import json
from .lowdim_dataset import LowDimDataset
from .lowdim_modeling import get_lowdim_attention


class LowDimTrainer:
    def __init__(self, model, lowdim_module_factory, tokenizer, 
                 num_train_instances, num_train_tokens_per_instance,
                 num_val_instances, num_val_tokens_per_instance,
                 epochs, train_batch_size=16, eval_batch_size=16,
                 lowdim_attn_layers = None):
        self.model = model
        self.lowdim_module_factory = lowdim_module_factory
        self.epochs = epochs
        self.lowdim_attentions = []
        self.tokenizer = tokenizer
        self.lowdim_attn_layers = lowdim_attn_layers
        if hasattr(self.model.config, "head_dim"):
            original_dim = self.model.config.head_dim
        else:
            original_dim = int(self.model.config.hidden_size / self.model.config.num_attention_heads)
        self.lowdim_module_factory.set_original_dim(original_dim)

        # Load and prepare LowDim datasets based on BookSum data
        self.train_dataset = LowDimDataset(self.tokenizer, num_train_instances, num_train_tokens_per_instance, split='train')
        self.val_dataset = LowDimDataset(self.tokenizer, num_val_instances, num_val_tokens_per_instance, split='validation')
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=eval_batch_size, shuffle=False)

        # Inject LowDim attentions
        attention_layer_counter = 0
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention) or isinstance(module, Qwen3Attention) or isinstance(module, Qwen2Attention):
                if self.lowdim_attn_layers is None or attention_layer_counter in self.lowdim_attn_layers:
                    new_attention = get_lowdim_attention(
                        module=module, 
                        lowdim_model=self.lowdim_module_factory.create()
                    )
                    self.lowdim_attentions.append(new_attention)

                    # Replace the module in its parent
                    parent_module = model.get_submodule(name.rsplit('.', 1)[0])
                    setattr(parent_module, name.rsplit('.', 1)[1], new_attention)
                attention_layer_counter += 1
        if attention_layer_counter == 0:
            raise Exception(f"{attention_layer_counter} attentions layers found! Only these architectures are supported: Llama3, Qwen2, Qwen3")

        logging.info("LowDimTrainer initialized.")

    def train(self):
        self._set_train_mode()
        for epoch in range(self.epochs):
            # train
            logging.info(f"LowDimTrainer training epoch {epoch+1} started.")
            for train_batch_id, batch in enumerate(self.train_loader):
                logging.info(f"Starting {train_batch_id+1} batch training.")
                self._generate(batch["input_ids"], batch["attention_mask"])
                self._finalize_epoch_training()
            logging.info(f"LowDimTrainer training epoch {epoch+1} finished.")

            # Log val
            self._start_score_collection()
            for val_batch in self.val_loader:
                self._generate(val_batch["input_ids"], val_batch["attention_mask"])
            self._finalize_score_collection()
            self._set_train_mode()

        self._set_eval_mode()

        logging.info(f"LowDimTrainer training finished.")

    def eval(self, num_instances, num_tokens_per_instance, batch_size=16, split="test"):
        # Prepare test LowDim data
        test_dataset = LowDimDataset(self.tokenizer, num_instances, num_tokens_per_instance, split=split)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Evaluate
        self._start_score_collection()
        for batch in test_loader:
                self._generate(batch["input_ids"], batch["attention_mask"])
        scores = self._finalize_score_collection()

        logging.info(f"LowDimTrainer evaluation finished.")
        return scores

    def save_lowdim_checkpoints(self, target_dir="./results/checkpoints"):
        for attention_layer in self.lowdim_attentions:
            attention_layer.save(target_dir)

    ######################
    ## HELPER FUNCTIONS ##
    ######################
    
    def _generate(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.7,
                top_k=50
            )

    def _set_train_mode(self):
        for attn in self.lowdim_attentions:
            attn.lowdim_train()

    def _finalize_epoch_training(self):
        for attn in self.lowdim_attentions:
            attn.finalize_lowdim_epoch_training()

    def _set_eval_mode(self):
        for attn in self.lowdim_attentions:
            attn.lowdim_eval()

    def _start_score_collection(self):
        self._set_eval_mode()
        for attn in self.lowdim_attentions:
            attn.start_score_collection()

    def _finalize_score_collection(self):
        scores_means = {}
        for i, attn in enumerate(self.lowdim_attentions):
            scores_means[i] = attn.finalize_score_collection()
        logging.info(json.dumps(scores_means, indent=4))
        return scores_means