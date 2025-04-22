def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    m, n = len(a), len(a[0])
    b = [[0] * m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            b[j][i] = a[i][j]
    return b

if __name__ == "__main__":
    a = [[1, 2, 3], [4, 5, 6]]
    print(transpose_matrix(a)) # [[1, 4], [2, 5], [3, 6]]
    a = [[1, 2], [3, 4]]
    print(transpose_matrix(a)) # [[1, 3], [2, 4]]
    a = [[1]]
    print(transpose_matrix(a)) # [[1]]