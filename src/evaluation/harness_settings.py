import torch
from torch import nn as nn
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from types import MethodType
from src.models.lowdim_modeling import AutoLowDimAttentionsModel, parse_lowdim_attn_layers
import time
import psutil
import os
import json


@register_model("LowDimAttentionsModel")
class LowDimEvalWrapper(HFLM):


    def __init__(self, lowdim_attentions_path, lowdim_attn_layers=None, monitoring_dir="./monitoring", **kwargs):
        super().__init__(**kwargs)

        assert self._model is not None, "Parent HFLM class did not initialize a model."
        assert self.tokenizer is not None, "Parent HFLM class did not initialize a tokenizer."
        self.tokenizer.pad_token = self.tokenizer.eos_token
        lowdim_attn_layers = parse_lowdim_attn_layers(lowdim_attn_layers)
        self._model = AutoLowDimAttentionsModel.inject_lowdim_attentions(
            model=self._model,
            lowdim_attentions_path=lowdim_attentions_path, 
            lowdim_attn_layers=lowdim_attn_layers
        )

        self._total_generated_tokens = 0
        self._total_eval_tokens = 0
        self._total_generation_time = 0
        self._peak_cpu_memory_mb = 0
        lowdim_attention_setting = lowdim_attentions_path.split("/")[-1] if len(lowdim_attentions_path.split("/")[-1]) != 0 else len(lowdim_attentions_path.split("/")[-2])
        lowdim_attention_setting += f"_{len(lowdim_attn_layers)}.json" if lowdim_attn_layers is not None else "_all.json"
        self._monitoring_output_json = os.path.join(monitoring_dir,lowdim_attention_setting)



    @torch.no_grad()
    def generate_until(self, requests):
        self._update_peak_cpu_memory()

        start_time = time.time()
        generations = super().generate_until(requests)
        end_time = time.time()
        
        generation_time = end_time - start_time
        tokens_in_batch = sum(len(self.tokenizer.encode(gen, add_special_tokens=False)) for gen in generations)
        
        self._total_generated_tokens += tokens_in_batch
        self._total_generation_time += generation_time
        self._update_peak_cpu_memory()
        
        print(f"Batch generation time: {generation_time:.2f}s, tokens: {tokens_in_batch}.")
        
        return generations

    @torch.no_grad()
    def loglikelihood(self, requests):
        self._update_peak_cpu_memory()

        start_time = time.time()
        loglikelihoods = super().loglikelihood(requests)
        end_time = time.time()
        
        tokens_in_batch = sum(
            len(self.tokenizer.encode(req.args[0], add_special_tokens=False)) + 
            len(self.tokenizer.encode(req.args[1], add_special_tokens=False)) 
            for req in requests
        )
        
        self._total_eval_tokens += tokens_in_batch
        self._total_generation_time += (end_time - start_time)
        
        self._update_peak_cpu_memory()
        
        return loglikelihoods

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        # The original code did not monitor this, so we add the logic here.
        self._update_peak_cpu_memory()
        
        start_time = time.time()
        loglikelihoods = super().loglikelihood_rolling(requests)
        end_time = time.time()

        # Count tokens for rolling loglikelihood
        tokens_in_batch = sum(len(self.tokenizer.encode(req.args[0], add_special_tokens=False)) for req in requests)
        self._total_eval_tokens += tokens_in_batch
        self._total_generation_time += (end_time - start_time)

        self._update_peak_cpu_memory()
        return loglikelihoods

    def _update_peak_cpu_memory(self):
        process = psutil.Process()
        current_rss = process.memory_info().rss / (1024**2)
        self._peak_cpu_memory_mb = max(self._peak_cpu_memory_mb, current_rss)


    def report_metrics(self):
        total_tokens = self._total_generated_tokens + self._total_eval_tokens
        metrics_data = {}

        if self._total_generation_time > 0 and total_tokens > 0:
            throughput = total_tokens / self._total_generation_time
            metrics_data.update({
                "total_tokens_processed": total_tokens,
                "total_processing_time_s": round(self._total_generation_time, 4),
                "average_throughput_tokens_per_s": round(throughput, 4),
                "peak_cpu_memory_mb": round(self._peak_cpu_memory_mb, 4)
            })

        if torch.cuda.is_available():
            total_gpu_peak_mb = 0
            for i in range(torch.cuda.device_count()):
                total_gpu_peak_mb += torch.cuda.max_memory_allocated(device=i) / (1024**2)
            metrics_data["total_peak_gpu_memory_across_all_gpus_mb"] = round(total_gpu_peak_mb, 4)
    
        print(metrics_data)
        # Save the dictionary to the specified JSON file
        with open(self._monitoring_output_json, "w") as f:
            json.dump(metrics_data, f, indent=4)



    def __del__(self):
        self.report_metrics()