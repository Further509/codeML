import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    gamma = np.array(gamma)
    beta = np.array(beta)
    
    B, C, H, W = X.shape
    group_size = C // num_groups
    X = X.reshape(B, num_groups, group_size, H, W)
    mean = np.mean(X, axis=(2, 3, 4), keepdims=True)
    var = np.var(X, axis=(2, 3, 4), keepdims=True)
    X = (X - mean) / np.sqrt(var + epsilon)

    gamma = gamma.reshape(-1, 1, 1, 1)
    beta = beta.reshape(-1, 1, 1, 1)

    X = gamma * X + beta
    X = X.reshape(B, C, H, W)

    return X

if __name__ == "__main__":
    X = np.random.randn(2, 2, 2, 2)
    gamma = [1, 1] 
    beta = [0, 0] 
    num_groups = 2
    ans = group_normalization(X, gamma, beta, num_groups)
    print(ans)