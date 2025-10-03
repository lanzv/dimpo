from torch.utils.data import Dataset
import torch
from datasets import load_dataset
import random

class LowDimDataset(Dataset):
    def __init__(self, tokenizer, num_instances, num_tokens_per_instance, split="train"):
        self.tokenizer = tokenizer
        self.num_instances = num_instances
        self.num_tokens_per_instance = num_tokens_per_instance

        dataset = load_dataset("kmfoda/booksum", split=split,  streaming=True)

        self.data = []
        token_count = 0
        
        for instance in dataset:
            chapter_text = instance["chapter"]
            tokens = tokenizer.encode(chapter_text, add_special_tokens=False)
            if len(tokens) >= num_tokens_per_instance:
                self.data.append(tokens[:num_tokens_per_instance])
                token_count += 1
            if token_count >= num_instances:
                break
        
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance_tokens = self.data[idx]
        input_ids = torch.tensor(instance_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }