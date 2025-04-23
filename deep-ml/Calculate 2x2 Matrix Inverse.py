def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    if a * d - b * c == 0:
        return None
    scaler = 1 / (a * d - b * c )
    inverse =[[d * scaler, -b * scaler], [-c * scaler, a * scaler]]
    return inverse

if __name__ == "__main__":
    matrix = [[4, 7], [2, 6]]
    inverse = inverse_2x2(matrix)
    if inverse:
        print(f"The inverse of the matrix is: {inverse}")
    else:
        print("The matrix is singular and cannot be inverted.")