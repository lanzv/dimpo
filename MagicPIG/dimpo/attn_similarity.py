import torch
import torch.nn.functional as F






def eval_attentions_kl_divergence(gold, pred, input_mask):
    """
    Computes the mean KL divergence, normalizing each individual score by 
    the number of unmasked keys used to compute it.

    Args:
        gold (torch.Tensor): The target or "true" attention distributions,
                             shape (batch, heads, n_queries, n_keys).
        pred (torch.Tensor): The predicted attention distributions,
                             shape (batch, heads, n_queries, n_keys).
        input_mask (torch.Tensor): The padding mask, shape (batch, n_keys).
                                  True for valid tokens, False for padding.

    Returns:
        list: A list of floats, where each float is the KL divergence for a 
              single attention distribution.
    """
    # Get original shapes
    assert gold.shape == pred.shape
    b, h, n_queries, n_keys = gold.shape

    # Combine batch, heads, and queries into a single dimension N
    gold = gold.view(-1, n_keys)  # Shape (N, n_keys)
    pred = pred.view(-1, n_keys)  # Shape (N, n_keys)

    # Apply the mask
    if input_mask is not None:
        mask_expanded = input_mask.unsqueeze(1).unsqueeze(1).expand(-1, h, n_queries, -1).reshape(-1, n_keys)
        gold = gold.masked_fill(mask_expanded == 0, 0)
        pred = pred.masked_fill(mask_expanded == 0, 0)
    
    # Compute KL divergence for each distribution
    gold = F.normalize(gold, p=1, dim=-1)
    pred = F.normalize(pred, p=1, dim=-1)    
    pred_log_prob = torch.log(pred + 1e-9)
    kl_divergences_all = F.kl_div(pred_log_prob, gold, reduction='none')
    kl_divergences_summed = kl_divergences_all.sum(dim=-1) # Shape (N,)

    # Normalize each score by its sequence length
    #valid_keys_count = mask_expanded.sum(dim=-1).clamp_min(1) if input_mask is not None else n_keys # Shape (N,)
    kl_divergences_normalized = kl_divergences_summed # / valid_keys_count

    # Compute the final mean and return as a float
    return kl_divergences_normalized.tolist()






def eval_attentions_js_divergence(gold, pred, input_mask):
    """Computes the JS divergence for attention distributions.

    Args:
        gold (torch.Tensor): The target attention distributions, shape (batch, heads, n_queries, n_keys).
        pred (torch.Tensor): The predicted attention distributions, shape (batch, heads, n_queries, n_keys).
        input_mask (torch.Tensor): Padding mask, shape (batch, n_keys).

    Returns:
        list: A list of floats, where each float is the normalized JS divergence for a single
              (batch, head, query) combination.
    """
    assert gold.shape == pred.shape
    b, h, n_queries, n_keys = gold.shape
    n = b * h * n_queries

    # Flatten tensors
    gold = gold.view(n, n_keys)
    pred = pred.view(n, n_keys)

    # Expand and apply mask
    if input_mask is not None:
        mask_expanded = input_mask.unsqueeze(1).unsqueeze(1).expand(-1, h, n_queries, -1).reshape(n, n_keys)
        gold = gold.masked_fill(mask_expanded == 0, 0)
        pred = pred.masked_fill(mask_expanded == 0, 0)
    
    # Normalize distributions
    gold = F.normalize(gold, p=1, dim=-1)
    pred = F.normalize(pred, p=1, dim=-1)
    
    # Average distribution for JSD
    m = 0.5 * (gold + pred)

    # Compute KL divergences
    log_m = torch.log(m + 1e-9)
    kl_gold_m = F.kl_div(log_m, gold, reduction='none').sum(dim=-1)
    kl_pred_m = F.kl_div(log_m, pred, reduction='none').sum(dim=-1)

    # Calculate JSD and normalize
    js_divergences_summed = 0.5 * (kl_gold_m + kl_pred_m)
    #valid_keys_count = mask_expanded.sum(dim=-1).clamp_min(1) if input_mask is not None else n_keys
    js_divergences_normalized = js_divergences_summed # / valid_keys_count

    return js_divergences_normalized.tolist()






def eval_attentions_mse(gold, pred, input_mask=None):
    """Computes the Mean Squared Error (MSE) for attention distributions.

    Args:
        gold (torch.Tensor): The target attention distributions, shape (batch, heads, n_queries, n_keys).
        pred (torch.Tensor): The predicted attention distributions, shape (batch, heads, n_queries, n_keys).
        input_mask (torch.Tensor): Padding mask, shape (batch, n_keys).

    Returns:
        list: A list of floats, where each float is the normalized MSE for a single
              (batch, head, query) combination.
    """
    assert gold.shape == pred.shape
    b, h, n_queries, n_keys = gold.shape
    n = b * h * n_queries

    # Flatten tensors
    gold = gold.view(n, n_keys)
    pred = pred.view(n, n_keys)

    # Expand and apply mask
    if input_mask is not None:
        mask_expanded = input_mask.unsqueeze(1).unsqueeze(1).expand(-1, h, n_queries, -1).reshape(n, n_keys)
        gold = gold.masked_fill(mask_expanded == 0, 0)
        pred = pred.masked_fill(mask_expanded == 0, 0)
    
    # Compute squared differences
    # MSE for each distribution is the sum of squared errors divided by the number of elements
    squared_diff = (gold - pred)**2

    # Sum across the `n_keys` dimension for each distribution
    mse_summed = squared_diff.sum(dim=-1)  # Shape (N,)

    # Normalize each score by its sequence length (number of valid keys)
    valid_keys_count = mask_expanded.sum(dim=-1).clamp_min(1) if input_mask is not None else n_keys
    mse_normalized = mse_summed / valid_keys_count
    
    return mse_normalized.tolist()






def eval_attentions_mae(gold, pred, input_mask):
    """Computes the Mean Absolute Error (MAE) for attention distributions.

    Args:
        gold (torch.Tensor): The target attention distributions, shape (batch, heads, n_queries, n_keys).
        pred (torch.Tensor): The predicted attention distributions, shape (batch, heads, n_queries, n_keys).
        input_mask (torch.Tensor): Padding mask, shape (batch, n_keys).

    Returns:
        list: A list of floats, where each float is the normalized MAE for a single
              (batch, head, query) combination.
    """
    assert gold.shape == pred.shape
    b, h, n_queries, n_keys = gold.shape
    n = b * h * n_queries

    # Flatten tensors
    gold = gold.view(n, n_keys)
    pred = pred.view(n, n_keys)

    # Expand and apply mask
    if input_mask is not None:
        mask_expanded = input_mask.unsqueeze(1).unsqueeze(1).expand(-1, h, n_queries, -1).reshape(n, n_keys)
        gold = gold.masked_fill(mask_expanded == 0, 0)
        pred = pred.masked_fill(mask_expanded == 0, 0)
    
    # Compute absolute differences
    absolute_diff = torch.abs(gold - pred)

    # Sum across the `n_keys` dimension for each distribution
    mae_summed = absolute_diff.sum(dim=-1)  # Shape (N,)

    # Normalize each score by its sequence length (number of valid keys)
    valid_keys_count = mask_expanded.sum(dim=-1).clamp_min(1) if input_mask is not None else n_keys
    mae_normalized = mae_summed / valid_keys_count
    
    return mae_normalized.tolist()






def compute_attention_score_distribution(queries, keys, input_mask):
    """
    Computes Grouped-Query Attention (GQA) with a padding mask.

    Args:
        queries (torch.Tensor): Query tensor with shape (b, num_query_heads, query_len, dim).
        keys (torch.Tensor): Key tensor with shape (b, num_key_heads, key_len, dim).
        input_mask (torch.Tensor): Padding mask tensor with shape (b, key_len).
            This mask indicates valid tokens (True) and padding tokens (False).

    Returns:
        torch.Tensor: The final attention score distribution with shape (b, num_query_heads, query_len, key_len).
    """
    b, num_key_heads, key_len, dim = keys.shape
    b_q, num_query_heads, query_len, dim_q = queries.shape
    assert b == b_q
    assert dim == dim_q

    # Handle GQA by repeating keys/values
    if num_query_heads != num_key_heads:
        assert num_query_heads > num_key_heads and num_query_heads % num_key_heads == 0
        reps = num_query_heads // num_key_heads
        keys = keys.unsqueeze(2).repeat(1, 1, reps, 1, 1).view(b, num_query_heads, key_len, dim)

    # Calculate the dot product attention scores
    dot = torch.einsum('bhid,bhjd->bhij', queries, keys) * (dim ** -0.5)

    # Apply the input mask
    if input_mask is not None:
        mask = input_mask.unsqueeze(1).unsqueeze(2).expand_as(dot)
        dot = dot.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to get the final distribution
    dot = dot.softmax(dim=-1)

    return dot


def compute_full_attention(values, softmax_dot):
    """
    Computes the full attention output given queries, keys, values, and an optional mask.

    Args:
        queries (torch.Tensor): Query tensor with shape (b, num_query_heads, query_len, dim).
        keys (torch.Tensor): Key tensor with shape (b, num_key_heads, key_len, dim).
        values (torch.Tensor): Value tensor with shape (b, num_key_heads, key_len, dim_v).
    Returns:
        torch.Tensor: The final attention output with shape (b, num_query_heads, query_len, dim_v).
    """
    b, num_value_heads, value_len, dim_v = values.shape
    b_q, num_query_heads, query_len, key_len = softmax_dot.shape
    assert b == b_q

    # Handle GQA for values by repeating
    if num_query_heads != num_value_heads:
        assert num_query_heads > num_value_heads and num_query_heads % num_value_heads == 0
        reps = num_query_heads // num_value_heads
        values = values.unsqueeze(2).repeat(1, 1, reps, 1, 1).view(b, num_query_heads, key_len, dim_v)

    # Multiply attention weights by values to get the final output
    output = torch.einsum('bhij,bhjd->bhid', softmax_dot, values)
    return output