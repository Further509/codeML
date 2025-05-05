import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    # Your code here
    return np.round(gamma * np.tanh(alpha * x) + beta, 4)

if __name__ == "__main__":
    x = np.array([[[0.14115588, 0.00372817, 0.24126647, 0.22183601]]])
    gamma = np.ones((4,))
    beta = np.zeros((4,))
    alpha = 0.5
    print(dynamic_tanh(x, alpha, gamma, beta))