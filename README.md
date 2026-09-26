# DimPO

Code for the paper *DimPO: Dimensionality Reduction for Attention using Preference Optimization*.

A linear map can send queries and keys of a frozen language model into a lower dimension without updating the model. The obvious training objective is to match the full attention distribution, for example by minimizing KL. DimPO asks a different question: whether a preference over keys, together with the attention mass on the keys the teacher weights most, is a better signal for keeping the model's behavior. The projection is trained on its own in every layer, so later layers cannot compensate for it. Values are never projected.

## Loss

The same map $F_\theta: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d'}$ is applied to a query and its keys. $\psi$ denotes the original attention weights. $p$ is the softmax of the projected scores over the full key list. Keys are sorted by $\psi$ descending, so the first $k$ positions are the head. The listwise term is a reference-free preference loss with margin $\gamma \ge 0$,

$$\mathcal{L}_{\mathrm{list}}(\pi_\theta) = \mathbb{E}_{(x,y,\psi)\sim \mathcal{D}} \left\lbrack \sum_{\psi_i > \psi_j} \Delta_{i,j}\, \log \left(1 + e^{-(s_i - s_j - \gamma)}\right) \right\rbrack,$$

$$\Delta_{i,j} = \lvert G_i - G_j \rvert \cdot \left\lvert \frac{1}{D(\tau(i))} - \frac{1}{D(\tau(j))} \right\rvert, \quad G_i = 2^{\psi_i} - 1, \quad D(\tau(i)) = \log(1+\tau(i)), \quad s_i = \beta \log \pi_\theta(y_i \mid x).$$

$\tau(i)$ is the rank of $y_i$ under the teacher weights $\psi$, with rank 1 for the largest $\psi_i$. The head term is a cross-entropy on those top-$k$ positions. The normalizer is still the full list,

$$\mathcal{L}_{\mathrm{head}} = - \sum_{i=1}^{k} \psi_i \log p_i .$$

DimPO is the sum

$$\mathcal{L}_{\mathrm{DimPO}} = \mathcal{L}_{\mathrm{list}} + \lambda \, \mathcal{L}_{\mathrm{head}}.$$

The default is $k=64$ and $\lambda=1$. Setting $k=0$ and $\lambda=0$ removes the head term and leaves the listwise objective alone.

## RULER

KL reconstructs the original attention more closely. DimPO keeps more of the downstream score once many layers are projected. On Llama3.1-8B, with $d'=64$, DimPO still holds about 95% of the original RULER 4k score at 50% of the layers. Further down, where the objectives separate:

| Method | Llama3.1-8B, $l=20$ | Qwen3-4B, $l=24$ |
| --- | ---: | ---: |
| Base ($l=0$) | 95.0 | 93.8 |
| DimPO ($k=0$, $\lambda=0$) | 35.6 | 63.4 |
| DimPO ($k=64$, $\lambda=1$) | **52.4** | **65.9** |
| KL | 38.9 | 50.2 |

RULER 4k average at $d'=64$. $l=20$ is 62.5% of the layers on Llama3.1-8B, $l=24$ is 66.7% on Qwen3-4B. Projections are trained on BookSum sequences of 4096 tokens.

## Setup

```sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

`torch==2.5.1+cu121` is the CUDA 12.1 wheel, so the PyTorch index is required. `evaluate_harness.py` needs `lm_eval==0.4.9.1` and registers the model name `LowDimAttentionsModel`.

## Checkpoints

`checkpoints/` has one directory per model, $d'=64$, $k=64$, $\lambda=1$, one file per layer:

- `DimPO_llama3_8b_instruct_64`
- `DimPO_llama3_3b_instruct_64`
- `DimPO_qwen3_4b_instruct_64`
- `DimPO_qwen2_7b_instruct_64`
- `DimPO_llama3_1b_instruct_64`

Llama3.2-1B already has head dimension 64, so that checkpoint is not a reduction.

## Load a projection

```python
from src.models.lowdim_modeling import AutoLowDimAttentionsModel
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
dimpo_path = "./checkpoints/DimPO_llama3_8b_instruct_64"
# last 8 layers of a 32-layer model
projected_attention_layers = [24, 25, 26, 27, 28, 29, 30, 31]

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoLowDimAttentionsModel.from_pretrained(
    model_path=model_id,
    lowdim_attentions_path=dimpo_path,
    device_map="cuda:0",
    lowdim_attn_layers=projected_attention_layers,
)
```

```python
prompt = "Explain why the sky is blue."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(inputs.input_ids, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Train

Local model directories are in `MODEL_PATHS` at the top of `run_projection_experiment.py`. Training reads BookSum from `../data/booksum`.

```sh
python3 run_projection_experiment.py \
    --model_name "llama3_8b_instruct" \
    --target_dim "64" \
    --projection_type "DimPO" \
    --checkpoint_path "./checkpoints/DimPO_llama3_8b_instruct_64" \
    --disable_evaluation
```

`--k 0` drops the head term. `--lmbda` sets $\lambda$.
