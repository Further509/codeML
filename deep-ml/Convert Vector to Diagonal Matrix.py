import numpy as np

def make_diagonal(x):
    # Your code here
    size = x.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        ans[i][i] = x[i]
    return ans

if __name__ == "__main__":
    x = np.array([1, 2, 3])
    output = make_diagonal(x)
    print(output)