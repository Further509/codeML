import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    n_batch, l_seq, d_model = x.shape

    gating_scores = np.einsum('bld,de->ble', x, Wg)

    top_k_indices = np.argpartition(gating_scores, -top_k, axis=-1)[..., -top_k:]
    top_k_scores = np.take_along_axis(gating_scores, top_k_indices, axis=-1)

    exp_scores = np.exp(top_k_scores)
    gating_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    output = np.zeros((n_batch, l_seq, d_model))

    for b in range(n_batch):
        for l in range(l_seq):
            for k in range(top_k):
                expert_index = top_k_indices[b, l, k]
                expert_weight = We[expert_index]
                expert_output = np.dot(x[b, l], expert_weight)
                output[b, l] += gating_probs[b, l, k] * expert_output

    return output

if __name__ == "__main__":
    x = np.arange(12).reshape(2, 3, 2)
    We = np.ones((4, 2, 2))
    Wg = np.ones((2, 4))
    top_k = 1
    n_experts = 4

    result = moe(x, We, Wg, n_experts, top_k)
    print(result)