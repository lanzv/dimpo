import atexit
import json
import os
import time

import psutil
import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from src.models.lowdim_modeling import AutoLowDimAttentionsModel, parse_lowdim_attn_layers


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
            lowdim_attn_layers=lowdim_attn_layers,
        )

        # Timing / throughput. For generate_until we count prompt + generated tokens
        # against wall-clock prefill+decode time (fair long-context tok/s).
        self._total_prompt_tokens = 0
        self._total_generated_tokens = 0
        self._total_eval_tokens = 0
        self._total_requests = 0
        self._total_generation_time = 0.0
        self._peak_cpu_memory_mb = 0.0
        self._gpu_peak_reset = False
        self._metrics_reported = False

        lowdim_attention_setting = (
            lowdim_attentions_path.split("/")[-1]
            if len(lowdim_attentions_path.split("/")[-1]) != 0
            else lowdim_attentions_path.split("/")[-2]
        )
        lowdim_attention_setting += (
            f"_{len(lowdim_attn_layers)}.json" if lowdim_attn_layers is not None else "_all.json"
        )
        os.makedirs(monitoring_dir, exist_ok=True)
        self._monitoring_output_json = os.path.join(monitoring_dir, lowdim_attention_setting)
        atexit.register(self.report_metrics)

    def _reset_gpu_peak_once(self):
        """Drop load-time allocations so peak reflects timed eval only."""
        if self._gpu_peak_reset or not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device=i)
        self._gpu_peak_reset = True

    def _count_prompt_tokens(self, requests):
        n = 0
        for req in requests:
            ctx = req.args[0]
            n += len(self.tokenizer.encode(ctx, add_special_tokens=False))
        return n

    @torch.no_grad()
    def generate_until(self, requests):
        self._reset_gpu_peak_once()
        self._update_peak_cpu_memory()

        prompt_tokens = self._count_prompt_tokens(requests)
        start_time = time.perf_counter()
        generations = super().generate_until(requests)
        generation_time = time.perf_counter() - start_time

        gen_tokens = sum(
            len(self.tokenizer.encode(gen, add_special_tokens=False)) for gen in generations
        )

        self._total_prompt_tokens += prompt_tokens
        self._total_generated_tokens += gen_tokens
        self._total_requests += len(requests)
        self._total_generation_time += generation_time
        self._update_peak_cpu_memory()

        print(
            f"generate_until wall={generation_time:.2f}s "
            f"requests={len(requests)} "
            f"prompt_tok={prompt_tokens} gen_tok={gen_tokens} "
            f"total_tok={prompt_tokens + gen_tokens}."
        )
        return generations

    @torch.no_grad()
    def loglikelihood(self, requests):
        self._reset_gpu_peak_once()
        self._update_peak_cpu_memory()

        start_time = time.perf_counter()
        loglikelihoods = super().loglikelihood(requests)
        elapsed = time.perf_counter() - start_time

        tokens_in_batch = sum(
            len(self.tokenizer.encode(req.args[0], add_special_tokens=False))
            + len(self.tokenizer.encode(req.args[1], add_special_tokens=False))
            for req in requests
        )

        self._total_eval_tokens += tokens_in_batch
        self._total_requests += len(requests)
        self._total_generation_time += elapsed
        self._update_peak_cpu_memory()
        return loglikelihoods

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        self._reset_gpu_peak_once()
        self._update_peak_cpu_memory()

        start_time = time.perf_counter()
        loglikelihoods = super().loglikelihood_rolling(requests)
        elapsed = time.perf_counter() - start_time

        tokens_in_batch = sum(
            len(self.tokenizer.encode(req.args[0], add_special_tokens=False)) for req in requests
        )
        self._total_eval_tokens += tokens_in_batch
        self._total_requests += len(requests)
        self._total_generation_time += elapsed
        self._update_peak_cpu_memory()
        return loglikelihoods

    def _update_peak_cpu_memory(self):
        process = psutil.Process()
        current_rss = process.memory_info().rss / (1024**2)
        self._peak_cpu_memory_mb = max(self._peak_cpu_memory_mb, current_rss)

    def report_metrics(self):
        if self._metrics_reported:
            return
        self._metrics_reported = True

        total_generate_tokens = self._total_prompt_tokens + self._total_generated_tokens
        total_tokens = total_generate_tokens + self._total_eval_tokens
        metrics_data = {
            "n_requests": self._total_requests,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_generated_tokens": self._total_generated_tokens,
            "total_eval_tokens": self._total_eval_tokens,
            "total_tokens_processed": total_tokens,
            "total_processing_time_s": round(self._total_generation_time, 4),
            "peak_cpu_memory_mb": round(self._peak_cpu_memory_mb, 4),
        }

        if self._total_generation_time > 0:
            if total_tokens > 0:
                metrics_data["average_throughput_tokens_per_s"] = round(
                    total_tokens / self._total_generation_time, 4
                )
            # Decode-only view (variable with answer length; keep for diagnostics).
            if self._total_generated_tokens > 0:
                metrics_data["average_generated_throughput_tokens_per_s"] = round(
                    self._total_generated_tokens / self._total_generation_time, 4
                )
            if self._total_requests > 0:
                metrics_data["average_latency_s_per_request"] = round(
                    self._total_generation_time / self._total_requests, 6
                )

        if torch.cuda.is_available():
            total_gpu_peak_mb = 0.0
            for i in range(torch.cuda.device_count()):
                total_gpu_peak_mb += torch.cuda.max_memory_allocated(device=i) / (1024**2)
            metrics_data["total_peak_gpu_memory_across_all_gpus_mb"] = round(total_gpu_peak_mb, 4)

        print(metrics_data)
        os.makedirs(os.path.dirname(self._monitoring_output_json) or ".", exist_ok=True)
        with open(self._monitoring_output_json, "w") as f:
            json.dump(metrics_data, f, indent=4)

    def __del__(self):
        try:
            self.report_metrics()
        except Exception:
            pass
