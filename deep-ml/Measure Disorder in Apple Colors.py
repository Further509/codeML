from collections import Counter
import math

def disorder(apples: list) -> float:
    """
    Compute the disorder in a basket of apples.
    """
    # Your code here
    cnt = Counter(apples)
    n = len(apples)
    entropy = 0
    for key, value in cnt.items():
        probability = value / n
        entropy -= probability * math.log2(probability)
    return entropy

if __name__ == "__main__":
    print(disorder([0, 0, 0, 0]))  # 输出 0
    print(disorder([1, 1, 0, 0]))  # 输出大于 0 的值
    print(disorder([0, 1, 2, 3]))  # 输出大于 [1, 1, 0, 0] 的值
    print(disorder([0, 0, 1, 1, 2, 2, 3, 3]))  # 输出大于 [0, 0, 0, 0, 0, 1, 2, 3] 的值