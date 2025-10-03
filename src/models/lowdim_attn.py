import torch
from torch import nn

class BaseLowDimAttention(nn.Module):
    """Base class for LowDim wrappers — no forward implementation here."""

    def __init__(self, base_attention, lowdim_model):
        super().__init__()
        self.config = base_attention.config
        self.layer_idx = base_attention.layer_idx
        self.head_dim = base_attention.head_dim
        self.num_key_value_groups = base_attention.num_key_value_groups
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = base_attention.attention_dropout
        self.is_causal = base_attention.is_causal

        # LowDim state
        self.lowdim_model = lowdim_model
        self.lowdim_train_mode = False
        self.lowdim_project = False
        self.lowdim_collect_scores = False

        # Original projections
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj

        # Model-specific extras (may be None for some models)
        self.q_norm = getattr(base_attention, "q_norm", None)
        self.k_norm = getattr(base_attention, "k_norm", None)
        self.sliding_window = getattr(base_attention, "sliding_window", None)


    ######################
    ## LowDim utilities ##
    ######################

    def lowdim_train(self):
        self.lowdim_train_mode = True
        self.lowdim_project = False

    def finalize_lowdim_epoch_training(self):
        self.lowdim_model.finalize_epoch_training()

    def lowdim_eval(self):
        self.lowdim_train_mode = False
        self.lowdim_project = True

    def start_score_collection(self):
        self.lowdim_project = False
        self.lowdim_collect_scores = True

    def finalize_score_collection(self):
        self.lowdim_project = True
        self.lowdim_collect_scores = False
        return self.lowdim_model.finalize_score_collection()

    def save(self, target_dir="./results/checkpoints"):
        self.lowdim_model.save(target_dir, self.layer_idx)
