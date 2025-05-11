
import numpy as np

def cosine_similarity(v1, v2):
    # Implement your code here
    dot = 0
    v12 = 0
    v22 = 0
    for x, y in zip(v1, v2):
        dot += x * y
        v12 += x ** 2
        v22 += y ** 2
    return round(dot / (np.sqrt(v12) * np.sqrt(v22)), 3)

if __name__ == "__main__":
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 4, 6])
    print(cosine_similarity(v1, v2))