import math

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    trace = matrix[0][0] + matrix[1][1]
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    temp = trace ** 2 - 4 * det
    temp = math.sqrt(temp)
    lam1 = (trace + temp) / 2
    lam2 = (trace - temp) / 2
    return [lam1, lam2]