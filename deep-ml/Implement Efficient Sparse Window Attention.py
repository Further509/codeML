import numpy as np
def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    # Your code here
    seq_len = Q.shape[0]
    d_k = Q.shape[1]
    d_v = V.shape[1]

    if scale_factor is None:
        scale_factor = np.sqrt(d_k)

    attention_results = np.zeros((seq_len, d_v))

    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)

        Q_window = Q[i]
        K_window = K[start: end]
        V_window = V[start: end]

        scores = np.dot(Q_window, K_window.T) / scale_factor
        weights = np.exp(scores) / np.sum(np.exp(scores)) # softmax
        weights_V = weights @ V_window
        attention_results[i] = weights_V

    return attention_results

if __name__ == "__main__":
    Q = np.array([[1.0], [1.0], [1.0]])
    K = np.array([[1.0], [1.0], [1.0]])
    V = np.array([[1.0], [2.0], [3.0]])
    print(sparse_window_attention(Q, K, V, 1))
    Q = np.array([[0.0], [1.0], [0.0],[2.0], [0.0], [7.0]]) 
    K = np.array([[1.0], [2.0], [3.0], [0.0], [6.0], [0.0]]) 
    V = np.array([[10.0], [20.0], [30.0],[12.0], [23.0], [70.0]]) 
    print(sparse_window_attention(Q, K, V, 2))