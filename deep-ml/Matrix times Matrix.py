def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    m, n = len(a), len(a[0])
    p, q = len(b), len(b[0])
    if n != p:
        return -1
    c = [[0] * q for _ in range(m)]
    for i in range(m):
        for k in range(q):
            for j in range(n):
                c[i][k] += a[i][j] * b[j][k]
    return c    

if __name__ == "__main__":
    A = [[1,2],[2,4]] 
    B = [[2,1],[3,4]]
    C = matrixmul(A, B)
    print(C)