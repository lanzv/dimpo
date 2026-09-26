import abc
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from ..evaluation.attn_similarity import (
    compute_attention_score_distribution,
    eval_attentions_kl_divergence,
    eval_attentions_js_divergence,
    eval_attentions_mse,
    eval_attentions_mae,
    compute_full_attention
)
from .dimpo_loss import DimPOLoss
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
        pass

    def partial_train(self, queries, keys):
        pass

    def finalize_epoch_training(self):
        pass

    def eval_batch(self, queries, keys, input_mask=None, values=None):
        original_attention = compute_attention_score_distribution(queries, keys, input_mask)

        # Project queries and keys using the abstract method
        projected_queries = self.project(queries)
        projected_keys = self.project(keys)

        projected_attention = compute_attention_score_distribution(projected_queries, projected_keys, input_mask)

        _, H, _, _ = original_attention.shape
        chunk_size_h = 4  # adjust if needed

        for h_start in range(0, H, chunk_size_h):
            h_end = min(h_start + chunk_size_h, H)

            # Slice chunk of heads
            orig_chunk = original_attention[:, h_start:h_end, :, :]
            proj_chunk = projected_attention[:, h_start:h_end, :, :]

            # Same slicing for mask
            mask_chunk = input_mask if input_mask is not None else None

            # Evaluate metrics on this head chunk
            self._scores["kl_divergence"] += eval_attentions_kl_divergence(
                orig_chunk, proj_chunk, mask_chunk
            )
            self._scores["js_divergence"] += eval_attentions_js_divergence(
                orig_chunk, proj_chunk, mask_chunk
            )
            self._scores["mse"] += eval_attentions_mse(
                orig_chunk, proj_chunk, mask_chunk
            )
            self._scores["mae"] += eval_attentions_mae(
                orig_chunk, proj_chunk, mask_chunk
            )

        if values is not None:
            original_fa = compute_full_attention(values=values, softmax_dot=original_attention)
            projected_fa = compute_full_attention(values=values, softmax_dot=projected_attention)
            _, H, _, _ = original_fa.shape
            for h_start in range(0, H, chunk_size_h):
                h_end = min(h_start + chunk_size_h, H)

                # Slice chunk of heads
                orig_chunk = original_fa[:, h_start:h_end, :, :]
                proj_chunk = projected_fa[:, h_start:h_end, :, :]

            self._scores["fa_mse"] += eval_attentions_mse(orig_chunk, proj_chunk)

    def finalize_score_collection(self):
        scores_means = {
            "kl_divergence": float(np.mean(self._scores["kl_divergence"])),
            "js_divergence": float(np.mean(self._scores["js_divergence"])),
            "mse": float(np.mean(self._scores["mse"])),
            "mae": float(np.mean(self._scores["mae"])),
            "fa_mse": float(np.mean(self._scores["fa_mse"])) if len(self._scores["fa_mse"]) > 0 else -1.0,
        }
        self._scores = {key: [] for key in self._scores}
        return scores_means
    
    def _sample_queries(self, queries):
        return queries[:, :, ::queries.shape[1], :]
    
    def save(self, target_dir, layer_id):
        logging.warning(f"Config might be incompleted.")
        checkpoint = {
            "model_state_dict": self.model.linear.state_dict(),
            "config": {
                "target_dim": self.target_dim,
                "original_dim": self.original_dim,
                "beta": None,
                "gamma": None,
                "lr": None,
                "batch_size": None
            }
        }
        path_to_checkpoint = os.path.join(target_dir, f"checkpoint_{layer_id}.pth")
        torch.save(checkpoint, path_to_checkpoint)






















################
## LowDim PCA ##
################
            
class LowDimPCAFactory(LowDimFactoryBase):
    def __init__(self, target_dim, full_queries=False):
        super().__init__()
        self.target_dim = target_dim
        self.full_queries = full_queries

    def create(self):
        super().create()
        return LowDimPCA(target_dim=self.target_dim, full_queries=self.full_queries)

class LowDimPCA(LowDimModuleBase):
    def __init__(self, target_dim, full_queries=False):
        super().__init__()
        self.target_dim = target_dim
        self.full_queries = full_queries
        self.pca = None
        self._collected_data = []

    def project(self, vectors):
        if self.pca is None:
            raise RuntimeError("PCA model has not been trained yet. Call finalize_epoch_training first.")
        original_shape = vectors.shape
        last_dim = original_shape[-1]
        flat_vectors = vectors.reshape(-1, last_dim)
        flat_vectors_np = flat_vectors.detach().cpu().numpy()
        projected_np = self.pca.transform(flat_vectors_np)
        projected_tensor = torch.tensor(projected_np, device=vectors.device, dtype=vectors.dtype)
        new_shape = original_shape[:-1] + (self.target_dim,)
        return projected_tensor.reshape(new_shape)

    def partial_train(self, queries, keys):
        if not self.full_queries:
            queries = self._sample_queries(queries)
        dim = queries.shape[-1]
        self._collected_data.append(queries.reshape(-1, dim).detach().cpu())
        self._collected_data.append(keys.reshape(-1, dim).detach().cpu())

    def finalize_epoch_training(self):
        all_vectors = torch.cat(self._collected_data, dim=0)
        self.pca = PCA(n_components=self.target_dim)
        self.pca.fit(all_vectors.numpy())
        self._collected_data = []






















##################
## LowDim DimPO ##
##################


class LowDimDimPOFactory(LowDimFactoryBase):
    def __init__(self, target_dim, beta=2.5, gamma=0.0, lr=0.01, batch_size=1, k=64, lmbda=1.0, num_sampled_keys=None):
        super().__init__()
        self.target_dim = target_dim
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.k = k
        self.lmbda = lmbda
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
            k=self.k,
            lmbda=self.lmbda,
            num_sampled_keys=self.num_sampled_keys
        )


class LowDimDimPO(LowDimModuleBase):
    def __init__(self, target_dim, original_dim, beta=2.5, gamma=0.0, lr=0.01, batch_size=1, k=64, lmbda=1.0, num_sampled_keys=None, loaded_state_dict=None, dtype=torch.float32):
        super().__init__()
        self.target_dim = target_dim
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        self.lmbda = lmbda
        self.original_dim = original_dim

        if loaded_state_dict is None:
            self.model = LinearProjection(original_dim, target_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.model = LinearProjection(original_dim, target_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.linear.load_state_dict(loaded_state_dict)
            self.model.to(dtype)
            self.optimizer = None
        self._collected_data_q = []
        self._collected_data_k = []
        self._collected_data_scores = []

        self.num_sampled_keys = num_sampled_keys
        self.skipping_not_logged_yet = True


        self.loss_fn = DimPOLoss(beta=self.beta, gamma=self.gamma, k=self.k, lmbda=self.lmbda)

        
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

        q_flat, k_flat, scores_flat = self._prepare_data_for_dimpo(queries, keys)
        
        self._collected_data_q.append(q_flat.detach().cpu())
        self._collected_data_k.append(k_flat.detach().cpu())
        self._collected_data_scores.append(scores_flat.detach().cpu())

        if len(self._collected_data_q) * q_flat.shape[0] >= self.batch_size:
            self._train_collected_batch()

    
    def finalize_epoch_training(self):
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
                "batch_size": self.batch_size,
                "k": self.k,
                "lmbda": self.lmbda,
            }
        }
        path_to_checkpoint = os.path.join(target_dir, f"checkpoint_{layer_idx}.pth")
        torch.save(checkpoint, path_to_checkpoint)

    def load_from_disk(target_dir, layer_idx, dtype=torch.float32):
        path_to_checkpoint = os.path.join(target_dir, f"checkpoint_{layer_idx}.pth")
        loaded_checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
        loaded_state_dict = loaded_checkpoint["model_state_dict"]
        config = loaded_checkpoint["config"]
        return LowDimDimPO(target_dim=config["target_dim"], original_dim=config["original_dim"], beta=config["beta"], gamma=config["gamma"], lr=config["lr"], batch_size=config["batch_size"], k=config.get("k", config.get("top_k", 0)), lmbda=config.get("lmbda", config.get("kl_weight", 0.0)), loaded_state_dict=loaded_state_dict, dtype=dtype)
    

    ############################
    ###    DimPO Training    ###
    ############################

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
            keys = keys.repeat_interleave(H_q // H_k, dim=1)
            B_k, H_k, K, D_k = keys.shape
    
        all_train_queries = []
        all_train_keys = []
        all_train_scores = []
    
        for h in range(H_q):
            head_queries = queries[:, h, :, :].unsqueeze(1).cpu()  # [B, 1, Q, D_q]
            head_keys = keys[:, h, :, :].unsqueeze(1).cpu()      # [B, 1, K, D_k]
    
            # Compute attention scores
            scores = torch.einsum('bhqd, bhkd -> bhqk', head_queries, head_keys)
            scores = scores / (D_q ** 0.5)
            scores = F.softmax(scores, dim=-1)
    
            # Sort by teacher attention, descending, so the head is the first k keys
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
                b_q, b_keys, b_scores = b_q.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), b_keys.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), b_scores.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                # Project
                proj_q = self.model(b_q)
                proj_q = proj_q.unsqueeze(1)
                proj_ks = self.model(b_keys.reshape(-1, b_keys.shape[-1])).reshape(b_keys.shape[0], b_keys.shape[1], -1)
                
                # L_DimPO = L_list + lambda * L_head
                loss = self.loss_fn(proj_q=proj_q, proj_ks=proj_ks, psi=b_scores)    # [B, K]
    
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.skipping_not_logged_yet:
                        self.skipping_not_logged_yet = False
                        logging.warning(f"Skipping update for batch {i} due to NaN/Inf loss")
                    continue
                loss.backward()
                self.optimizer.step()

        # Clear processed data, keeping any remainder.
        # Clone so an empty/short remainder does not retain the full parent storage.
        processed_count = (num_samples // batch_size) * batch_size
        self._collected_data_q = [all_q[processed_count:].clone()]
        self._collected_data_k = [all_k[processed_count:].clone()]
        self._collected_data_scores = [all_scores[processed_count:].clone()]
        self.model.eval()



















#################
## LowDim Rand ##
#################


class LowDimRandFactory(LowDimFactoryBase):
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def create(self):
        super().create()
        return LowDimRand(original_dim=self.original_dim, target_dim=self.target_dim)


class LowDimRand(LowDimModuleBase):

    def __init__(self, target_dim, original_dim):
        super().__init__()
        self.target_dim = target_dim
        self.original_dim = original_dim
        self.random_matrix = torch.randn(original_dim, target_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def project(self, vectors):
        original_shape = vectors.shape
        last_dim = original_shape[-1]
        if last_dim != self.original_dim:
            raise ValueError(f"Input vector dimension mismatch. Expected {self.original_dim}, got {last_dim}.")
        flat_vectors = vectors.reshape(-1, last_dim)
        model_device = self.random_matrix.device
        vector_device = flat_vectors.device
        projected_tensor = torch.matmul(flat_vectors.to(model_device), self.random_matrix)
        new_shape = original_shape[:-1] + (self.target_dim,)
        return projected_tensor.reshape(new_shape).to(vector_device)

    def partial_train(self, queries, keys):
        pass

    def finalize_epoch_training(self):
        pass






















##################
## LowDim SimPO ##
##################


class LowDimSimPOFactory(LowDimFactoryBase):
    def __init__(self, target_dim, beta=2.5, gamma=0.0, lr=0.01, batch_size=1,
                experimental_mode="none", num_sampled_keys=None,
                key_vector_pair_distance=None, num_sampled_pairs=None):
        super().__init__()
        self.target_dim = target_dim
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.experimental_mode=experimental_mode 
        self.num_sampled_keys=num_sampled_keys
        self.key_vector_pair_distance=key_vector_pair_distance
        self.num_sampled_pairs=num_sampled_pairs

    def create(self):
        super().create()
        return LowDimSimPO(
            target_dim=self.target_dim,
            original_dim=self.original_dim,
            beta=self.beta,
            gamma=self.gamma,
            lr=self.lr,
            batch_size=self.batch_size,
            experimental_mode=self.experimental_mode, 
            num_sampled_keys=self.num_sampled_keys,
            key_vector_pair_distance=self.key_vector_pair_distance,
            num_sampled_pairs=self.num_sampled_pairs
        )


class LowDimSimPO(LowDimModuleBase):
    def __init__(self, target_dim, original_dim, beta=2.0, gamma=0.5, lr=0.01, batch_size=1,
                experimental_mode="none", num_sampled_keys=None, key_vector_pair_distance=None,
                num_sampled_pairs=None):
        super().__init__()
        self.target_dim = target_dim
        self.original_dim = original_dim
        self.beta = beta
        self.gamma = gamma # Target reward margin for SimPO
        self.batch_size = batch_size

        self.model = LinearProjection(original_dim, target_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # SimPO collects pairs of (chosen, rejected) keys for each query.
        self._collected_data_q = []
        self._collected_data_chosen_k = []
        self._collected_data_rejected_k = []

        self.experimental_mode = experimental_mode
        self.num_sampled_keys = num_sampled_keys
        self.key_vector_pair_distance = key_vector_pair_distance
        self.num_sampled_pairs = num_sampled_pairs

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
        queries = self._sample_queries(queries)
        # Prepare SimPO data by finding the best and worst keys for each query
        if self.experimental_mode == "projection_performance":
            q_flat, chosen_k_flat, rejected_k_flat = self._prepare_data_for_simpo(queries, keys)
        else:
            q_flat, chosen_k_flat, rejected_k_flat = self._prepare_data_for_sampled_experimetns(queries, keys)

        self._collected_data_q.append(q_flat.detach().cpu())
        self._collected_data_chosen_k.append(chosen_k_flat.detach().cpu())
        self._collected_data_rejected_k.append(rejected_k_flat.detach().cpu())

        # Train the model if enough data is collected
        if len(self._collected_data_q) * q_flat.shape[0] >= self.batch_size:
            self._train_collected_batch()

    def finalize_epoch_training(self):
        # Train on any remaining data
        self._train_collected_batch()

    #############################
    ###    SimPO Training     ###
    #############################

    def _prepare_data_for_simpo(self, queries, keys):
        B_q, H_q, Q, D_q = queries.shape
        B_k, H_k, K, D_k = keys.shape
    
        # Handle Grouped-Query Attention (GQA) where H_q != H_k
        if H_q != H_k:
            keys = keys.repeat_interleave(H_q // H_k, dim=1)
            B_k, H_k, K, D_k = keys.shape
    
        all_train_queries = []
        all_train_chosen_keys = []
        all_train_rejected_keys = []
    
        for h in range(H_q):
            head_queries = queries[:, h, :, :]  # [B, Q, D_q]
            head_keys = keys[:, h, :, :]        # [B, K, D_k]
    
            # Compute attention scores
            scores = torch.einsum('bqd,bkd->bqk', head_queries, head_keys)
            
            # Find the index of the best and worst keys for each query
            _, best_indices = torch.max(scores, dim=-1) # [B, Q]
            _, worst_indices = torch.min(scores, dim=-1) # [B, Q]

            # Gather the best and worst keys
            chosen_keys = torch.gather(
                head_keys.unsqueeze(1).expand(-1, Q, -1, -1),
                dim=2,
                index=best_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, D_k)
            ).squeeze(2) # [B, Q, D_k]

            rejected_keys = torch.gather(
                head_keys.unsqueeze(1).expand(-1, Q, -1, -1),
                dim=2,
                index=worst_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, D_k)
            ).squeeze(2) # [B, Q, D_k]
            
            all_train_queries.append(head_queries.reshape(-1, D_q).cpu())
            all_train_chosen_keys.append(chosen_keys.reshape(-1, D_k).cpu())
            all_train_rejected_keys.append(rejected_keys.reshape(-1, D_k).cpu())
    
        train_queries = torch.cat(all_train_queries, dim=0)
        train_chosen_keys = torch.cat(all_train_chosen_keys, dim=0)
        train_rejected_keys = torch.cat(all_train_rejected_keys, dim=0)
    
        return train_queries, train_chosen_keys, train_rejected_keys

    def _train_collected_batch(self):
        all_q = torch.cat(self._collected_data_q, dim=0)
        all_chosen_k = torch.cat(self._collected_data_chosen_k, dim=0)
        all_rejected_k = torch.cat(self._collected_data_rejected_k, dim=0)
        
        num_samples = all_q.shape[0]
        batch_size = self.batch_size
        if num_samples < self.batch_size:
            batch_size = num_samples
            if batch_size == 0:
                return

        dataset = TensorDataset(all_q, all_chosen_k, all_rejected_k)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.enable_grad():
            self.model.train()
            for batch_idx, (b_q, b_chosen_k, b_rejected_k) in enumerate(loader):
                self.optimizer.zero_grad()
                b_q, b_chosen_k, b_rejected_k = b_q.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), b_chosen_k.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), b_rejected_k.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                # Project
                proj_q = self.model(b_q)
                proj_chosen_k = self.model(b_chosen_k)
                proj_rejected_k = self.model(b_rejected_k)

                # Compute SimPO loss
                loss = self._compute_loss(proj_q, proj_chosen_k, proj_rejected_k)
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.skipping_not_logged_yet:
                        self.skipping_not_logged_yet = False
                        logging.warning(f"Skipping update for batch {batch_idx} due to NaN/Inf loss")
                    continue

                loss.backward()
                self.optimizer.step()

        # Clear processed data, keeping any remainder.
        # Clone so an empty/short remainder does not retain the full parent storage.
        processed_count = (num_samples // batch_size) * batch_size
        self._collected_data_q = [all_q[processed_count:].clone()]
        self._collected_data_chosen_k = [all_chosen_k[processed_count:].clone()]
        self._collected_data_rejected_k = [all_rejected_k[processed_count:].clone()]
        self.model.eval()


    def _compute_loss(self, proj_q, proj_chosen_k, proj_rejected_k):
        # Compute similarities
        sim_win = torch.sum(proj_q * proj_chosen_k, dim=-1)  # [B]
        sim_lose = torch.sum(proj_q * proj_rejected_k, dim=-1)  # [B]
        logits = torch.stack([sim_win, sim_lose], dim=-1)  # [B, 2]
        scale = 1.0 / (proj_q.shape[-1] ** 0.5)
        probs = torch.softmax(logits * scale, dim=-1) # [B, 2]
        prob_w = torch.clamp(probs[:, 0], EPSILON_FOR_LOGARITHMS, 1.0 - EPSILON_FOR_LOGARITHMS)
        prob_l = torch.clamp(probs[:, 1], EPSILON_FOR_LOGARITHMS, 1.0 - EPSILON_FOR_LOGARITHMS)

        # SimPO loss
        score = (self.beta / 1.0) * torch.log(prob_w) - (self.beta / 1.0) * torch.log(prob_l) - self.gamma
        loss = -torch.log(torch.sigmoid(score)).mean()  # scalar
        return loss

    def _prepare_data_for_sampled_experimetns(self, queries, keys):
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
        all_train_chosen_keys = []
        all_train_rejected_keys = []

        chosen_indices_map = torch.cat([torch.full((K-i-1,), i) for i in range(K-1)]).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B_q, 1, Q, -1)
        rejected_indices_map = torch.cat([torch.arange(i+1, K) for i in range(K-1)]).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B_q, 1, Q, -1)


        for h in range(H_q):
            # Extract the current head's query and key tensors
            head_queries = queries[:, h, :, :].unsqueeze(1).cpu()  # [B, 1, Q, D_q]
            head_keys = keys[:, h, :, :].unsqueeze(1).cpu()      # [B, 1, K, D_k]

            # Compute attention scores
            scores = torch.einsum('bhqd, bhkd -> bhqk', head_queries, head_keys)

            # Sort the attention scores and get sorted indices
            _, sorted_indices = torch.sort(scores, dim=-1, descending=True) # [B, 1, Q, K]

            if self.experimental_mode == "selection_all_key_pairs" or self.experimental_mode == "selection_multiple_distinct_pairs":
                if self.experimental_mode == "selection_multiple_distinct_pairs":
                    chosen_range_size = K // 2
                    rejected_range_size = K - chosen_range_size
                    chosen_indices = torch.rand(B_q, Q, chosen_range_size, device=sorted_indices.device).argsort(dim=-1)[:, :, :self.num_sampled_pairs]
                    rejected_indices = torch.rand(B_q, Q, rejected_range_size, device=sorted_indices.device).argsort(dim=-1)[:, :, :self.num_sampled_pairs] + chosen_range_size
                    chosen_indices_map = chosen_indices.unsqueeze(1)
                    rejected_indices_map = rejected_indices.unsqueeze(1)
                chosen_key_indices = torch.gather(sorted_indices, dim=-1, index=chosen_indices_map)
                rejected_key_indices = torch.gather(sorted_indices, dim=-1, index=rejected_indices_map)
                expanded_head_keys = head_keys.unsqueeze(2).expand(-1, -1, Q, -1, -1)
                chosen_keys = torch.gather(
                    expanded_head_keys, 
                    dim=3, 
                    index=chosen_key_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D_k)
                )
                rejected_keys = torch.gather(
                    expanded_head_keys, 
                    dim=3, 
                    index=rejected_key_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D_k)
                )
                # Expand queries to match the number of sampled pairs
                head_queries_expanded = head_queries.unsqueeze(3).expand(-1, -1, -1, chosen_keys.shape[-2], -1)
                # Flatten the results for concatenation later
                all_train_queries.append(head_queries_expanded.reshape(-1, D_q).cpu())
                all_train_chosen_keys.append(chosen_keys.reshape(-1, D_k).cpu())
                all_train_rejected_keys.append(rejected_keys.reshape(-1, D_k).cpu())

            elif self.experimental_mode == "selection_level_of_key_diversity":
                high = K - min(self.key_vector_pair_distance, K-1)
                chosen_indices = torch.randint(low=0, high=high, size=(sorted_indices.shape[0], Q), device=sorted_indices.device)

                chosen_indices = chosen_indices.unsqueeze(1).unsqueeze(-1)
                rejected_indices = chosen_indices + min(self.key_vector_pair_distance, K-1)
                chosen_key_indices = torch.gather(sorted_indices, dim=-1, index=chosen_indices)
                rejected_key_indices = torch.gather(sorted_indices, dim=-1, index=rejected_indices)

                # Gather the key vectors corresponding to these specific indices.
                chosen_keys = torch.gather(head_keys, dim=2, index=chosen_key_indices.expand(-1, -1, -1, D_k)) # [B, 1, Q, dim]
                rejected_keys = torch.gather(head_keys, dim=2, index=rejected_key_indices.expand(-1, -1, -1, D_k))  # [B, 1, Q, dim]
        
                # Queries remain the same, as there is only one chosen/rejected pair per query.
                all_train_queries.append(head_queries.reshape(-1, D_q).cpu())
                all_train_chosen_keys.append(chosen_keys.reshape(-1, D_k).cpu())
                all_train_rejected_keys.append(rejected_keys.reshape(-1, D_k).cpu())
            else:
                raise Exception(f"Not supported experimental mode {self.experimental_mode}")

        train_queries = torch.cat(all_train_queries, dim=0)
        train_chosen_keys = torch.cat(all_train_chosen_keys, dim=0)
        train_rejected_keys = torch.cat(all_train_rejected_keys, dim=0)

        return train_queries, train_chosen_keys, train_rejected_keys




















####################
## LowDim Triplet ##
####################


class LowDimTripletFactory(LowDimFactoryBase):
    def __init__(self, target_dim, margin=0.0, lr=0.01, batch_size=1,
                experimental_mode="none", num_sampled_keys=None,
                key_vector_pair_distance=None, num_sampled_pairs=None):
        super().__init__()
        self.target_dim = target_dim
        self.margin = margin
        self.lr = lr
        self.batch_size = batch_size
        self.experimental_mode=experimental_mode 
        self.num_sampled_keys=num_sampled_keys
        self.key_vector_pair_distance=key_vector_pair_distance
        self.num_sampled_pairs=num_sampled_pairs

    def create(self):
        super().create()
        return LowDimTriplet(
            target_dim=self.target_dim,
            original_dim=self.original_dim,
            margin=self.margin,
            lr=self.lr,
            batch_size=self.batch_size,
            experimental_mode=self.experimental_mode, 
            num_sampled_keys=self.num_sampled_keys,
            key_vector_pair_distance=self.key_vector_pair_distance,
            num_sampled_pairs=self.num_sampled_pairs
        )


class LowDimTriplet(LowDimSimPO):
    def __init__(self, target_dim, original_dim, margin=0.5, lr=0.01, batch_size=1,
                experimental_mode="none", num_sampled_keys=None,
                key_vector_pair_distance=None, num_sampled_pairs=None):
        super().__init__(
            target_dim=target_dim,
            original_dim=original_dim,
            beta=0,  # beta is not used in triplet loss
            gamma=margin, # The margin for triplet loss
            lr=lr,
            batch_size=batch_size,
            experimental_mode=experimental_mode,
            num_sampled_keys=num_sampled_keys,
            key_vector_pair_distance=key_vector_pair_distance,
            num_sampled_pairs=num_sampled_pairs
        )
        self.margin = self.gamma
        self.criterion = nn.TripletMarginLoss(margin=margin, p=2)  # Euclidean distance

    def _compute_loss(self, proj_q, proj_chosen_k, proj_rejected_k):
        return self.criterion(proj_q, proj_chosen_k, proj_rejected_k)












#################
## LowDim ORPO ##
#################


class LowDimORPOFactory(LowDimFactoryBase):
    def __init__(self, target_dim, lmbda=1.0, lr=0.01, batch_size=1,
                experimental_mode="none", num_sampled_keys=None,
                key_vector_pair_distance=None, num_sampled_pairs=None):
        super().__init__()
        self.target_dim = target_dim
        self.lmbda = lmbda
        self.lr = lr
        self.batch_size = batch_size
        self.experimental_mode=experimental_mode 
        self.num_sampled_keys=num_sampled_keys
        self.key_vector_pair_distance=key_vector_pair_distance
        self.num_sampled_pairs=num_sampled_pairs

    def create(self):
        super().create()
        return LowDimORPO(
            target_dim=self.target_dim,
            original_dim=self.original_dim,
            lmbda=self.lmbda,
            lr=self.lr,
            batch_size=self.batch_size,
            experimental_mode=self.experimental_mode, 
            num_sampled_keys=self.num_sampled_keys,
            key_vector_pair_distance=self.key_vector_pair_distance,
            num_sampled_pairs=self.num_sampled_pairs
        )


class LowDimORPO(LowDimSimPO):

    def __init__(self, target_dim: int, original_dim: int, lmbda: float = 0.1, lr: float = 0.01, batch_size: int = 1,
                experimental_mode="none", num_sampled_keys=None, key_vector_pair_distance=None, num_sampled_pairs=None):
        super().__init__(
            target_dim=target_dim,
            original_dim=original_dim,
            beta=lmbda,
            lr=lr,
            batch_size=batch_size,
            experimental_mode=experimental_mode,
            num_sampled_keys=num_sampled_keys,
            key_vector_pair_distance=key_vector_pair_distance,
            num_sampled_pairs=num_sampled_pairs
        )
        self.lmbda = lmbda

    def _compute_loss(self, proj_q, proj_chosen_k, proj_rejected_k):
        # Compute similarities
        sim_win = torch.sum(proj_q * proj_chosen_k, dim=-1)  # [B]
        sim_lose = torch.sum(proj_q * proj_rejected_k, dim=-1)  # [B]
        logits = torch.stack([sim_win, sim_lose], dim=-1)  # [B, 2]
        scale = 1.0 / (proj_q.shape[-1] ** 0.5)
        probs = torch.softmax(logits * scale, dim=-1) # [B, 2]
        prob_w = torch.clamp(probs[:, 0], EPSILON_FOR_LOGARITHMS, 1.0 - EPSILON_FOR_LOGARITHMS)
        prob_l = torch.clamp(probs[:, 1], EPSILON_FOR_LOGARITHMS, 1.0 - EPSILON_FOR_LOGARITHMS)

        # ORPO loss
        log_prob_w = torch.log(prob_w)
        log_prob_l = torch.log(prob_l)
        pw = torch.clamp(torch.exp((1.0 / 1.0) * log_prob_w), EPSILON_FOR_LOGARITHMS, 1.0 - EPSILON_FOR_LOGARITHMS) # 1/|y| = 1/1
        pl = torch.clamp(torch.exp((1.0 / 1.0) * log_prob_l), EPSILON_FOR_LOGARITHMS, 1.0 - EPSILON_FOR_LOGARITHMS) # 1/|y| = 1/1
        logit_pw = torch.log(pw / (1.0 - pw))
        logit_pl = torch.log(pl / (1.0 - pl))
        loss = -torch.log(pw) - self.lmbda * F.logsigmoid(logit_pw - logit_pl)
        return loss.mean()












################
## LowDim CPO ##
################


class LowDimCPOFactory(LowDimFactoryBase):
    def __init__(self, target_dim, beta=1.0, lmbda=0.5, lr=0.01, batch_size=1,
                experimental_mode="none", num_sampled_keys=None,
                key_vector_pair_distance=None, num_sampled_pairs=None):
        super().__init__()
        self.target_dim = target_dim
        self.beta = beta
        self.lr = lr
        self.batch_size = batch_size
        self.lmbda = lmbda
        self.experimental_mode=experimental_mode 
        self.num_sampled_keys=num_sampled_keys
        self.key_vector_pair_distance=key_vector_pair_distance
        self.num_sampled_pairs=num_sampled_pairs

    def create(self):
        super().create()
        return LowDimCPO(
            target_dim=self.target_dim,
            original_dim=self.original_dim,
            beta=self.beta,
            lr=self.lr,
            batch_size=self.batch_size, 
            lmbda=self.lmbda,
            experimental_mode=self.experimental_mode, 
            num_sampled_keys=self.num_sampled_keys,
            key_vector_pair_distance=self.key_vector_pair_distance,
            num_sampled_pairs=self.num_sampled_pairs
        )


class LowDimCPO(LowDimSimPO):

    def __init__(self, target_dim: int, original_dim: int, beta: float = 1.0, lmbda: float = 0.5, lr: float = 0.01, batch_size: int = 1,
                experimental_mode="none", num_sampled_keys=None, key_vector_pair_distance=None, num_sampled_pairs=None):
        super().__init__(
            target_dim=target_dim,
            original_dim=original_dim,
            beta=beta,
            gamma=lmbda,
            lr=lr,
            batch_size=batch_size,
            experimental_mode=experimental_mode,
            num_sampled_keys=num_sampled_keys,
            key_vector_pair_distance=key_vector_pair_distance,
            num_sampled_pairs=num_sampled_pairs
        )
        self.lmbda = lmbda

    def _compute_loss(self, proj_q, proj_chosen_k, proj_rejected_k):
        # Compute probs
        sim_win = torch.sum(proj_q * proj_chosen_k, dim=-1)  # [B]
        sim_lose = torch.sum(proj_q * proj_rejected_k, dim=-1)  # [B]
        logits = torch.stack([sim_win, sim_lose], dim=-1)  # [B, 2]
        scale = 1.0 / (proj_q.shape[-1] ** 0.5)
        probs = torch.softmax(logits * scale, dim=-1) # [B, 2]
        prob_w = torch.clamp(probs[:, 0], EPSILON_FOR_LOGARITHMS, 1.0 - EPSILON_FOR_LOGARITHMS)
        prob_l = torch.clamp(probs[:, 1], EPSILON_FOR_LOGARITHMS, 1.0 - EPSILON_FOR_LOGARITHMS)


        # CPO loss
        loss = -torch.log(torch.clamp(self.beta * torch.log(prob_w) - self.beta * torch.log(prob_l), min=EPSILON_FOR_LOGARITHMS)) - self.lmbda * torch.log(prob_w)
        return loss.mean()





















####################
## LowDim KL      ##
####################


class LowDimKLFactory(LowDimFactoryBase):
    def __init__(self, target_dim, lr=0.01, batch_size=1, num_sampled_keys=None):
        super().__init__()
        self.target_dim = target_dim
        self.lr = lr
        self.batch_size = batch_size
        self.num_sampled_keys = num_sampled_keys

    def create(self):
        super().create()
        return LowDimKL(
            target_dim=self.target_dim,
            original_dim=self.original_dim,
            lr=self.lr,
            batch_size=self.batch_size,
            num_sampled_keys=self.num_sampled_keys
        )

class LowDimKL(LowDimDimPO):
    def __init__(self, target_dim, original_dim, lr=0.01, batch_size=1,
                 num_sampled_keys=None, loaded_state_dict=None, dtype=torch.float32):
        super().__init__(
            target_dim=target_dim,
            original_dim=original_dim,
            lr=lr,
            batch_size=batch_size,
            k=0,
            lmbda=0.0,
            num_sampled_keys=num_sampled_keys,
            loaded_state_dict=loaded_state_dict,
            dtype=dtype
        )
        self.loss_fn = self._kl_loss

    def _kl_loss(self, proj_q, proj_ks, psi):
        # Predicted attention distribution from compressed Q/K
        dot = torch.einsum('bqe,bke->bqk', proj_q, proj_ks) * (proj_q.shape[-1] ** -0.5)  # [B, 1, K]
        pred_log_probs = torch.log_softmax(dot.squeeze(1), dim=-1)  # [B, K]
        # psi is the original (full-dim) softmax attention distribution
        true_probs = psi.clamp(min=EPSILON_FOR_LOGARITHMS)
        # KL(true || pred), same direction as eval_attentions_kl_divergence
        return F.kl_div(pred_log_probs, true_probs, reduction='batchmean')
