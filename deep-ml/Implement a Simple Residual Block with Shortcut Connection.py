import numpy as np

def relu(x):
    return np.maximum(x, 0.0)

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    # Your code here
    out1 = relu(np.dot(w1, x))
    out2 = np.dot(w2, out1)
    out3 = out2 + x
    out = relu(out3)
    return out

if __name__ == "__main__":
    x = np.array([1.0, 2.0])
    w1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    w2 = np.array([[0.5, 0.0], [0.0, 0.5]])
    print(residual_block(x, w1, w2))