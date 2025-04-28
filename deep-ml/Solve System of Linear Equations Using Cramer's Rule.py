import numpy as np
from copy import deepcopy

def cramers_rule(A, b):
    # Your code here
    detA = np.linalg.det(A)
    if detA == 0:
        return -1
    n = len(A)
    x = []
    for j in range(n):
        temp = deepcopy(A)
        for i in range(n):
            temp[i][j] = b[i]
        detT = np.linalg.det(temp)
        x.append(round(detT / detA, 4))
    return x

if __name__ == "__main__":
    A = [[2, -1, 3], [4, 2, 1], [-6, 1, -2]] 
    b = [5, 10, -3]
    ans = cramers_rule(A, b)
    print(ans)