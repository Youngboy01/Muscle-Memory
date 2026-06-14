def matrixmul(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]]:
    m = len(a)
    n = len(a[0])
    o = len(b)
    p = len(b[0])
    if n != o:
        return -1
    c = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(o):
                c[i][j] += a[i][k] * b[k][j]

    return c
