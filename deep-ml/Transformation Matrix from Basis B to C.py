import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B = np.array(B)
    C = np.array(C)
    P = B @ np.linalg.inv(C)
    return np.round(P, 4).tolist()

if __name__ == "__main__":
    B = [[1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1]]
    C = [[1, 2.3, 3], 
        [4.4, 25, 6], 
        [7.4, 8, 9]]
    ans = transform_basis(B, C)
    print(ans)