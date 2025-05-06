import numpy as np

def softplus(x):
    return np.log(1 + np.exp(x))

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    logits = np.dot(X, W_g)
    noise = np.dot(X, W_noise) 
    noisy_logits = logits + N * softplus(noise)

    topk_indices = np.argpartition(noisy_logits, -k, axis=1)[:, -k:]

    final_logits = np.full_like(noisy_logits, -np.inf)
    batch_size = X.shape[0]
    for i in range(batch_size):
        final_logits[i, topk_indices[i]] = noisy_logits[i, topk_indices[i]]
    exp_logits = np.exp(final_logits)
    gating_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return gating_probs

if __name__ == "__main__":
    X = np.array([[1.0, 2.0]]) 
    W_g = np.array([[1.0, 0.0], [0.0, 1.0]]) 
    W_noise = np.array([[0.5, 0.5], [0.5, 0.5]]) 
    N = np.array([[1.0, -1.0]]) 
    print(np.round(noisy_topk_gating(X, W_g, W_noise, N, k=2), 4))