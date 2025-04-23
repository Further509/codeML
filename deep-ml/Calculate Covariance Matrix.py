def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
    m, n = len(vectors), len(vectors[0])
    # 归一化
    for i in range(m):
        mean = sum(vectors[i]) / n
        for j in range(n):
            vectors[i][j] -= mean
    ans = [[0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            # 第 i 和 j 个特征的协方差
            total = 0
            for k in range(n):
                total += vectors[i][k] * vectors[j][k]
            ans[i][j] = total / (n - 1)
    return ans

if __name__ == "__main__":
    # Example usage
    vectors = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    covariance_matrix = calculate_covariance_matrix(vectors)
    print(covariance_matrix)
    vectors = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0]
    ]
    covariance_matrix = calculate_covariance_matrix(vectors)
    print(covariance_matrix)