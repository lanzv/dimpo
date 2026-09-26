"""Ablation: listwise term plus a full KL on the attention distribution.

DimPO itself is L_list + lambda * L_head and lives in DimPOLoss.
This file is only the extra KL term on top of L_list (k=0, lambda=0).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dimpo_loss import DimPOLoss
from .lowdim_modules import EPSILON_FOR_LOGARITHMS, LowDimDimPO, LowDimFactoryBase


class DimPOKLLoss(nn.Module):
    def __init__(self, beta: float = 1.0, gamma: float = 0.0001, kl_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.list_loss = DimPOLoss(beta=beta, gamma=gamma, k=0, lmbda=0.0, eps=eps)
        self.kl_weight = kl_weight
        self.eps = eps

    def forward(self, proj_q, proj_ks, psi):
        L_list = self.list_loss(proj_q, proj_ks, psi)
        dot = torch.einsum("bqe,bke->bqk", proj_q, proj_ks) * (proj_q.shape[-1] ** -0.5)
        pred_log_probs = torch.log_softmax(dot.squeeze(1), dim=-1)
        true_probs = psi.clamp(min=max(self.eps, EPSILON_FOR_LOGARITHMS))
        kl_loss = F.kl_div(pred_log_probs, true_probs, reduction="batchmean")
        return L_list + self.kl_weight * kl_loss


class LowDimDimPOKLFactory(LowDimFactoryBase):
    def __init__(
        self,
        target_dim,
        beta=1.0,
        gamma=0.0001,
        lr=0.0001,
        batch_size=1,
        kl_weight=1.0,
        num_sampled_keys=None,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.num_sampled_keys = num_sampled_keys

    def create(self):
        super().create()
        return LowDimDimPOKL(
            target_dim=self.target_dim,
            original_dim=self.original_dim,
            beta=self.beta,
            gamma=self.gamma,
            lr=self.lr,
            batch_size=self.batch_size,
            kl_weight=self.kl_weight,
            num_sampled_keys=self.num_sampled_keys,
        )


class LowDimDimPOKL(LowDimDimPO):
    def __init__(
        self,
        target_dim,
        original_dim,
        beta=1.0,
        gamma=0.0001,
        lr=0.0001,
        batch_size=1,
        kl_weight=1.0,
        num_sampled_keys=None,
        loaded_state_dict=None,
        dtype=torch.float32,
    ):
        super().__init__(
            target_dim=target_dim,
            original_dim=original_dim,
            beta=beta,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            k=0,
            lmbda=0.0,
            num_sampled_keys=num_sampled_keys,
            loaded_state_dict=loaded_state_dict,
            dtype=dtype,
        )
        self.kl_weight = kl_weight
        self.loss_fn = DimPOKLLoss(beta=self.beta, gamma=self.gamma, kl_weight=kl_weight)

    def save(self, target_dir, layer_idx):
        checkpoint = {
            "model_state_dict": self.model.linear.state_dict(),
            "config": {
                "target_dim": self.target_dim,
                "original_dim": self.original_dim,
                "beta": self.beta,
                "gamma": self.gamma,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "kl_weight": self.kl_weight,
            },
        }
        path_to_checkpoint = os.path.join(target_dir, f"checkpoint_{layer_idx}.pth")
        torch.save(checkpoint, path_to_checkpoint)
