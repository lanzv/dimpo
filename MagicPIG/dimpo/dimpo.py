import abc
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .attn_similarity import (
    compute_attention_score_distribution,
    eval_attentions_kl_divergence,
    eval_attentions_js_divergence,
    eval_attentions_mse,
    eval_attentions_mae,
    compute_full_attention
)
import logging
import os


EPSILON_FOR_LOGARITHMS = 1e-6

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)









#################
## LowDim Base ##
#################


class LowDimFactoryBase(abc.ABC):
    def __init__(self):
        self.original_dim = None

    def set_original_dim(self, original_dim):
        self.original_dim = original_dim

    def create(self):
        if self.original_dim is None:
            raise Exception("Original dimension is not set in LowDimFactory!")

class LowDimModuleBase(abc.ABC):
    """Abstract base class for models that evaluate attention similarity.

    Concrete subclasses must implement the 'project' method.
    """
    def __init__(self):
        self._scores = {
            "kl_divergence": [],
            "js_divergence": [],
            "mse": [],
            "fa_mse": [],
            "mae": []
        }
        self.supported_save = False

    @abc.abstractmethod
    def project(self, vectors):
        """Projects a vector into a lower-dimensional space.

        Args:
            vectors (torch.Tensor): The input vectors.

        Returns:
            torch.Tensor: The projected vectors.
        """
        pass

    def partial_train(self, queries, keys):
        """Perform a single training step.

        Concrete classes must implement this method if they are trainable.
        """
        pass

    def finalize_epoch_training(self):
        """Finalize training at the end of an epoch.

        Concrete classes can override this method if needed.
        """
        pass

    def eval_batch(self, queries, keys, input_mask=None, values=None):
        """Evaluates a single batch and collects similarity scores.

        Args:
            queries (torch.Tensor): The query vectors.
            keys (torch.Tensor): The key vectors.
            input_mask (torch.Tensor): The padding mask.
        """
        original_attention = compute_attention_score_distribution(queries, keys, input_mask)
        
        # Project queries and keys using the abstract method
        projected_queries = self.project(queries)
        projected_keys = self.project(keys)
        
        projected_attention = compute_attention_score_distribution(projected_queries, projected_keys, input_mask)
        
        self._scores["kl_divergence"] += eval_attentions_kl_divergence(original_attention, projected_attention, input_mask)
        self._scores["js_divergence"] += eval_attentions_js_divergence(original_attention, projected_attention, input_mask)
        self._scores["mse"] += eval_attentions_mse(original_attention, projected_attention, input_mask)
        self._scores["mae"] += eval_attentions_mae(original_attention, projected_attention, input_mask)
        if values != None:
            original_fa = compute_full_attention(values=values, softmax_dot=original_attention)
            projected_fa = compute_full_attention(values=values, softmax_dot=projected_attention)
            self._scores["fa_mse"] += eval_attentions_mse(original_fa, projected_fa)

    def finalize_score_collection(self):
        """Calculates and prints the mean of collected scores, then resets them."""
        scores_means = {
            "kl_divergence": float(np.mean(self._scores["kl_divergence"])),
            "js_divergence": float(np.mean(self._scores["js_divergence"])),
            "mse": float(np.mean(self._scores["mse"])),
            "mae": float(np.mean(self._scores["mae"])),
            "fa_mse": float(np.mean(self._scores["fa_mse"])) if len(self._scores["fa_mse"]) > 0 else -1.0,
        }

        # Reset scores for the next evaluation run
        self._scores = {key: [] for key in self._scores}
        return scores_means
    
    def _sample_queries(self, queries):
        return queries[:, :, ::queries.shape[1], :]
    
    def save(self, target_dir, layer_id):
        logging.warning(f"It is not supported to save weights.")


















##################
## LowDim DimPO ##
##################


class LowDimDimPOFactory(LowDimFactoryBase):
    def __init__(self, target_dim, beta=2.5, gamma=0.0, lr=0.01, batch_size=1, num_sampled_keys=None):
        super().__init__()
        self.target_dim = target_dim
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.num_sampled_keys = num_sampled_keys

    def create(self):
        super().create()
        return LowDimDimPO(
            target_dim=self.target_dim,
            original_dim=self.original_dim,
            beta=self.beta,
            gamma=self.gamma,
            lr=self.lr,
            batch_size=self.batch_size,
            num_sampled_keys=self.num_sampled_keys
        )


class LowDimDimPO(LowDimModuleBase):
    """
    Concrete implementation of LowDimModuleBase using DimPO-style training.
    """
    def __init__(self, target_dim, original_dim, beta=2.5, gamma=0.0, lr=0.01, batch_size=1, num_sampled_keys=None, loaded_state_dict=None, dtype=torch.float32):
        super().__init__()
        self.target_dim = target_dim
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.original_dim = original_dim

        if loaded_state_dict is None:
            self.model = LinearProjection(original_dim, target_dim).cuda()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.model = LinearProjection(original_dim, target_dim).cuda()
            self.model.linear.load_state_dict(loaded_state_dict)
            self.model.to(dtype)
            self.optimizer = None
        self._collected_data_q = []
        self._collected_data_k = []
        self._collected_data_scores = []

        self.num_sampled_keys = num_sampled_keys
        self.skipping_not_logged_yet = True

        
    def project(self, vectors):
        original_shape = vectors.shape
        last_dim = original_shape[-1]
        flat_vectors = vectors.reshape(-1, last_dim)
        vec_device = flat_vectors.device
        model_device = next(self.model.parameters()).device
        flat_vectors = flat_vectors.to(model_device)
        
        with torch.no_grad():
            proj = self.model(flat_vectors)
            
        new_shape = original_shape[:-1] + (self.target_dim,)
        return proj.reshape(new_shape).to(vec_device)


    def partial_train(self, queries, keys):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        queries = self._sample_queries(queries)
        # Prepare DimPO data from queries and keys
        q_flat, k_flat, scores_flat = self._prepare_data_for_dimpo(queries, keys)
        
        self._collected_data_q.append(q_flat.detach().cpu())
        self._collected_data_k.append(k_flat.detach().cpu())
        self._collected_data_scores.append(scores_flat.detach().cpu())

        # If enough data is collected, train the model
        if len(self._collected_data_q) * q_flat.shape[0] >= self.batch_size:
            self._train_collected_batch()

    
    def finalize_epoch_training(self):
        # Train on any remaining data
        self._train_collected_batch()

    def save(self, target_dir, layer_idx):
        checkpoint = {
            "model_state_dict": self.model.linear.state_dict(),
            "config": {
                "target_dim": self.target_dim,
                "original_dim": self.original_dim,
                "beta": self.beta, 
                "gamma": self.gamma,
                "lr": self.lr,
                "batch_size": self.batch_size
            }
        }
        path_to_checkpoint = os.path.join(target_dir, f"checkpoint_{layer_idx}.pth")
        torch.save(checkpoint, path_to_checkpoint)

    def load_from_disk(target_dir, layer_idx, dtype=torch.float32):
        path_to_checkpoint = os.path.join(target_dir, f"checkpoint_{layer_idx}.pth")
        loaded_checkpoint = torch.load(path_to_checkpoint)
        loaded_state_dict = loaded_checkpoint["model_state_dict"]
        config = loaded_checkpoint["config"]
        return LowDimDimPO(target_dim=config["target_dim"], original_dim=config["original_dim"], beta=config["beta"], gamma=config["gamma"], lr=config["lr"], batch_size=config["batch_size"], loaded_state_dict=loaded_state_dict, dtype=dtype)
    

    ###############################
    ###    DimPO Training     ###
    ###############################

    def _prepare_data_for_dimpo(self, queries, keys):
        if self.num_sampled_keys is not None:
            B_k, H_k, K, D_k = keys.shape
            keys = torch.gather(
                keys,
                dim=2,
                index=torch.rand(B_k, H_k, K, device=keys.device)
                    .argsort(dim=-1)[..., :self.num_sampled_keys]
                    .unsqueeze(-1)
                    .expand(-1, -1, -1, D_k)
            )
        B_q, H_q, Q, D_q = queries.shape
        B_k, H_k, K, D_k = keys.shape
    
        # Handle Grouped-Query Attention (GQA) where H_q != H_k
        if H_q != H_k:
            # Repeat key heads to match the number of query heads
            keys = keys.repeat_interleave(H_q // H_k, dim=1)
            B_k, H_k, K, D_k = keys.shape
    
        # Initialize lists to store results for each head
        all_train_queries = []
        all_train_keys = []
        all_train_scores = []
    
        for h in range(H_q):
            # Extract the current head's query and key tensors
            head_queries = queries[:, h, :, :].unsqueeze(1).cpu()  # [B, 1, Q, D_q]
            head_keys = keys[:, h, :, :].unsqueeze(1).cpu()      # [B, 1, K, D_k]
    
            # Compute attention scores
            scores = torch.einsum('bhqd, bhkd -> bhqk', head_queries, head_keys)
            scores = scores / (D_q ** 0.5)
            scores = F.softmax(scores, dim=-1)
    
            # Sort the attention scores and get sorted indices
            sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
            keys_reshaped = head_keys.reshape(B_k, K, D_k)
            sorted_indices_reshaped = sorted_indices.reshape(B_q, Q, K)
            gathered_keys = torch.gather(
                keys_reshaped.unsqueeze(1).expand(-1, Q, -1, -1),
                dim=2,
                index=sorted_indices_reshaped.unsqueeze(-1).expand(-1, -1, -1, D_k)
            )
            
            # Flatten and append to the lists
            all_train_queries.append(head_queries.reshape(-1, D_q).cpu())
            all_train_keys.append(gathered_keys.reshape(-1, K, D_k).cpu())
            all_train_scores.append(sorted_scores.reshape(-1, K).cpu())
    
        # Concatenate the results from all heads
        train_queries = torch.cat(all_train_queries, dim=0)
        train_keys = torch.cat(all_train_keys, dim=0)
        train_scores = torch.cat(all_train_scores, dim=0)

        return train_queries, train_keys, train_scores

    
    def _train_collected_batch(self):
        all_q = torch.cat(self._collected_data_q, dim=0)
        all_k = torch.cat(self._collected_data_k, dim=0)
        all_scores = torch.cat(self._collected_data_scores, dim=0)

        # Ensure we have at least one batch
        num_samples = all_q.shape[0]
        batch_size = self.batch_size
        if num_samples < self.batch_size:
            batch_size = num_samples
            if batch_size == 0:
                return

        dataset = TensorDataset(all_q, all_k, all_scores)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.enable_grad():
            self.model.train()
            for i, (b_q, b_keys, b_scores) in enumerate(loader):
                self.optimizer.zero_grad()
                b_q, b_keys, b_scores = b_q.cuda(), b_keys.cuda(), b_scores.cuda()

                # Project
                proj_q = self.model(b_q)
                proj_q = proj_q.unsqueeze(1)
                proj_ks = self.model(b_keys.reshape(-1, b_keys.shape[-1])).reshape(b_keys.shape[0], b_keys.shape[1], -1)
                
                # Compute Attention and Loss
                dot = torch.einsum('bqe,bke->bqk', proj_q, proj_ks) * (self.target_dim ** -0.5) # [B, 1, K]
                dot = dot.squeeze(1) # [B, K]
                probs = dot.softmax(dim=-1)  # [B, K]
                
                # DimPO loss
                logp = torch.log(torch.clamp(probs, EPSILON_FOR_LOGARITHMS))  # (B, K)
                psi  = b_scores   # (B, K)
                loss = self._dimpo_loss(logp, b_scores)
    
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.skipping_not_logged_yet:
                        self.skipping_not_logged_yet = False
                        logging.warning(f"Skipping update for batch {i} due to NaN/Inf loss")
                    continue
                loss.backward()
                self.optimizer.step()

        # Clear processed data, keeping any remainder
        processed_count = (num_samples // batch_size) * batch_size
        self._collected_data_q = [all_q[processed_count:]]
        self._collected_data_k = [all_k[processed_count:]]
        self._collected_data_scores = [all_scores[processed_count:]]
        self.model.eval()

    
    def _dimpo_loss(self, logp, psi):
        # Model scores s_i = β/|y_i| * log π_θ(y_i|x)    where |y_i|=1 since the probability is computed base on one softmax value only
        s = (self.beta / 1.0) * logp                             # [B, K]

        # Get ranks τ(i) from s (1 = best rank)
        order = torch.argsort(psi, dim=-1, descending=True)
        ranks = torch.empty_like(order, dtype=torch.long)
        arange = torch.arange(psi.size(1), device=psi.device).unsqueeze(0).expand_as(order)
        ranks.scatter_(1, order, arange + 1)        # [B, K]

        # Gains G_i = 2^{ψ_i} - 1
        G = torch.pow(2.0, psi) - 1.0               # [B, K]

        # Discounts D(τ(i)) = log(1 + τ(i))
        D = torch.log1p(ranks.float())              # [B, K]
        
        # Pairwise differences
        s_diff = s.unsqueeze(2) - s.unsqueeze(1) - self.gamma  # SimPO equation with  [B, K, K]
        psi_i = psi.unsqueeze(2)
        psi_j = psi.unsqueeze(1)
        mask = psi_i > psi_j                        # only pairs where ψ_i > ψ_j [B, K, K]

        # Lambda weights Δ_{i,j}
        G_i = G.unsqueeze(2)
        G_j = G.unsqueeze(1)
        delta_G = torch.abs(G_i - G_j) # [B, K, K]
        
        D_i = D.unsqueeze(2)
        D_j = D.unsqueeze(1)
        delta = delta_G * (1.0 / D_i - 1.0 / D_j) # [B, K, K]

        # Pairwise logistic loss 
        # SimPO adaptation log(1 + exp(-(β/|y_i| * log π_θ(y_i|x) - β/|y_j| * log π_θ(y_j|x) - γ))) 
        #     from original LiPO log(1 + exp(-(β * log (π_θ(y_i|x)/π_ref(y_i|x)) - β * log (π_θ(y_j|x)/π_ref(y_j|x))))) 
        pair_loss = F.softplus(-s_diff)             # log(1 + exp(-(s_i - s_j - γ)))  [B, K, K]

        # Apply mask and weight
        weighted = (delta * pair_loss) * mask                    # [B, K, K]
        loss_per_list = weighted.sum(dim=(1,2)) / mask.sum(dim=(1,2)).clamp_min(1)     #[B]
        
        return loss_per_list.mean()



