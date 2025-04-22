def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    m, n = len(matrix), len(matrix[0])
    for i in range(m):
        for j in range(n):
            matrix[i][j] *= scalar
    return matrix   

if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    scalar = 2
    result = scalar_multiply(matrix, scalar)
    print(result)  # Output: [[2, 4, 6], [8, 10, 12], [14, 16, 18]]