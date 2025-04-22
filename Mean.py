def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    m, n = len(matrix), len(matrix[0])
    if mode == "column":
        means = [0] * n
        for j in range(n):
            total = 0
            for i in range(m):
                total += matrix[i][j]
            means[j] = total / m
    elif mode == "row":
        means = [0] * m
        for i in range(m):
            total = 0
            for j in range(n):
                total += matrix[i][j]
            means[i] = total / n
    return means

if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(calculate_matrix_mean(matrix, "column")) # [4.0, 5.0, 6.0]
    print(calculate_matrix_mean(matrix, "row")) # [2.0, 5.0, 8.0]
    matrix = [[1, 2], [3, 4]]
    print(calculate_matrix_mean(matrix, "column")) # [2.0, 3.0]
    print(calculate_matrix_mean(matrix, "row")) # [1.5, 3.5]