def OSA(source: str, target: str) -> int:
    # Your code here
    m, n = len(source), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # 增加字符
    for j in range(n + 1): 
        dp[0][j] = j
    # 删除字符
    for i in range(m + 1):
        dp[i][0] = i
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
            # 交换相邻字符
            if i >= 2 and j >= 2 and source[i - 2] == target[j - 1] and source[i - 1] == target[j - 2]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

    return dp[m][n]

if __name__ == "__main__":
    source = "caper" 
    target = "acer" 
    output = OSA(source, target) 
    print(output)
    source = "telescope" 
    target = "microscope"
    output = OSA(source, target) 
    print(output)
    source = "london" 
    target = "paris"
    output = OSA(source, target) 
    print(output)