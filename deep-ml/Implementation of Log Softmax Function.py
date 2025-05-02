import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    # Your code here
    ans = np.array(scores)
    de = np.log(np.sum(np.exp(ans)))
    return np.round(ans - de, 4)

if __name__ == "__main__":
    A = np.array([1, 2, 3])
    print(log_softmax(A))