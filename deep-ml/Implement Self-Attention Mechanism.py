import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def softmax(X):
    exp_X = np.exp(X)
    total = np.sum(exp_X, axis=-1, keepdims=True)
    return exp_X / total

def self_attention(Q, K, V):
    d = Q.shape[-1]
    attention_output = softmax(Q @ K.T / np.sqrt(d)) @ V
    return attention_output

if __name__ == "__main__":
    X = np.array([[1, 0], [0, 1]])
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])

    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    output = self_attention(Q, K, V)

    print(output)