import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
        data (list[float]): A list of numerical values to transform.
        degree (int): The degree of the polynomial expansion.

    """
    # Your code here
    if degree < 0:
        return []
    m = len(data)
    ans = [[1.0] * (degree + 1) for _ in range(m)]
    for i in range(m):
        temp = data[i]
        for j in range(1, degree + 1):
            ans[i][j] = temp ** j
    return ans