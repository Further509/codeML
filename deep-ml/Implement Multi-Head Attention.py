import numpy as np

def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def compute_qkv(X, W_q, W_k, W_v):
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def self_attention(Q, K, V):
    d = Q.shape[-1]
    attention_output = softmax(Q @ K.T / np.sqrt(d)) @ V
    return attention_output

def multi_head_attention(Q, K, V, n_heads):
    d = Q.shape[-1]
    d_k = d // n_heads
    
    Q_head = np.split(Q, n_heads, axis=-1)
    K_head = np.split(K, n_heads, axis=-1)
    V_head = np.split(V, n_heads, axis=-1)
    attention_outputs = []
    for i in range(n_heads):
        attention_output = self_attention(Q_head[i], K_head[i], V_head[i])
        attention_outputs.append(attention_output)
    multi_head_output = np.concatenate(attention_outputs, axis=-1)
    return multi_head_output

if __name__ == "__main__":
    Q = np.array([[1, 0], [0, 1]])
    K = np.array([[1, 0], [0, 1]])
    V = np.array([[1, 0], [0, 1]])
    n_heads = 2
    ans = multi_head_attention(Q, K, V, n_heads)
    print(ans)