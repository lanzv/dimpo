import torch
import torch.nn as nn
import torch.nn.functional as F

class DimPOLoss(nn.Module):
    def __init__(self, beta: float = 2.5, gamma: float = 0.0, k: int = 64, lmbda: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.lmbda = lmbda
        self.eps = eps

    def forward(self, proj_q, proj_ks, psi):
        """
        proj_q:  [B, 1, D] projected queries
        proj_ks: [B, K, D] projected keys
        psi:     [B, K]    teacher attention weights, sorted descending
        """
        # Compute attention for logp of model
        dot = torch.einsum('bqe,bke->bqk', proj_q, proj_ks) * (proj_q.shape[-1] ** -0.5) # [B, 1, K]
        dot = dot.squeeze(1) # [B, K]
        p = dot.softmax(dim=-1)  # [B, K]  softmax over the full key list
        logp = torch.log(torch.clamp(p, min=self.eps))  # [B, K]

        # Compute predicted ranking model scores s_i = Beta * log pi_theta(y_i|x)
        s = self.beta * logp

        # tau(i): rank under the teacher weights psi (1 = largest psi)
        order = torch.argsort(psi, dim=-1, descending=True)
        ranks = torch.empty_like(order, dtype=torch.long)
        arange = torch.arange(psi.size(1), device=psi.device).unsqueeze(0).expand_as(order)
        ranks.scatter_(1, order, arange + 1)  

        # Gains G_i = 2^{psi_i} - 1
        G = torch.pow(2.0, psi) - 1.0
 
        # Discounts D(tau(i)) = log(1 + tau(i))
        D = torch.log1p(ranks.float())              # [B, K]
        
        # Pairwise diffs
        s_diff = s.unsqueeze(2) - s.unsqueeze(1) - self.gamma # margin equation  [B, K, K]
        mask = psi.unsqueeze(2) > psi.unsqueeze(1) # only pairs where psi_i > psi_j [B, K, K]

        # Delta_{i,j}
        delta_G = torch.abs(G.unsqueeze(2) - G.unsqueeze(1)) # [B, K, K]
        delta = delta_G * (1.0 / D.unsqueeze(2) - 1.0 / D.unsqueeze(1)) # [B, K, K]

        # Pairwise logistic loss 
        # SimPO adaptation log(1 + exp(-(beta * log pi_theta(y_i|x) - beta * log pi_theta(y_j|x) - gamma))) 
        #     from original LiPO log(1 + exp(-(beta * log (pi_theta(y_i|x)/pi_ref(y_i|x)) - beta * log (pi_theta(y_j|x)/pi_ref(y_j|x))))) 
        pair_loss = F.softplus(-s_diff)             # log(1 + exp(-(s_i - s_j - gamma)))  [B, K, K]

        # Apply mask and weight
        weighted = (delta * pair_loss) * mask                    # [B, K, K]
        loss_per_list = weighted.sum(dim=(1,2)) / mask.sum(dim=(1,2)).clamp_min(1)     #[B]
        L_list = loss_per_list.mean()

        # L_head = -sum_{i=1}^{k} psi_i log p_i, normalizer still over the full list.
        # k = 0 or lambda = 0 drops the head and leaves the listwise term.
        if self.k <= 0 or self.lmbda == 0.0:
            return L_list
        k = min(self.k, psi.shape[-1])
        log_p = torch.log_softmax(dot, dim=-1)
        L_head = -(psi[:, :k] * log_p[:, :k]).sum(dim=-1).mean()
        return L_list + self.lmbda * L_head