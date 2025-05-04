import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A_size = len(A)
    x = [0] * A_size
    for _ in range(n):
        x_prev = x.copy()
        for i in range(A_size):
            temp = 0
            for j in range(A_size):
                temp += A[i][j] * x_prev[j] if i != j else 0
            x[i] = round(1 / A[i][i] * (b[i] - temp), 4)
    return x

if __name__ == "__main__":
    A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]] 
    b = [-1, 2, 3] 
    n=2
    ans = solve_jacobi(A, b, n)
    print(ans)