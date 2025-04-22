import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    m, n = len(a), len(a[0])
    if m * n != new_shape[0] * new_shape[1]:
        return []
    matrix = [[0] * new_shape[1] for _ in range(new_shape[0])]
    for i in range(m * n):
        matrix[i // new_shape[1]][i % new_shape[1]] = a[i // n][i % n] 
    return matrix  

if __name__ == "__main__":
    a = [[1, 2, 3, 4], [5, 6, 7, 8]]
    new_shape = (4, 2)
    print(reshape_matrix(a, new_shape))
    a = [[1, 2, 3], [4, 5, 6]]
    new_shape = (3, 2)
    print(reshape_matrix(a, new_shape)) # [[1, 2], [3, 4], [5, 6]]
    a = [[1, 2], [3, 4]]
    new_shape = (4, 1)
    print(reshape_matrix(a, new_shape)) # [[1], [2], [3], [4]]
    a = [[1]]
    new_shape = (1, 1)
    print(reshape_matrix(a, new_shape)) # [[1]]