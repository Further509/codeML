import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    # Your code here
    H, W, C = x.shape
    inputs = x.transpose(2, 0, 1) 
    ans = np.zeros((C, ))
    for i in range(C):
        ans[i] = np.mean(inputs[i])

    return ans

if __name__ == "__main__":
    x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    ans = global_avg_pool(x)
    print(ans)