import numpy as np

def softmax(values):
    max_x = np.max(values, axis=-1, keepdims=True)
    exp_x = np.exp(values - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def pattern_weaver(n, crystal_values, dimension):
    # Your code here
    crystal_values = np.array(crystal_values).reshape(-1, dimension)
    attention_scores = np.dot(crystal_values, crystal_values.T)
    weighted_pattern = []
    for i in range(n):
        scores = softmax(attention_scores[i])
        weighted = np.sum(crystal_values * scores.reshape(-1, 1))
        weighted_pattern.append(weighted)
    return np.round(np.array(weighted_pattern), 3)

if __name__ == "__main__":
    ans = pattern_weaver(5, [4, 2, 7, 1, 9], 1)
    print(ans)