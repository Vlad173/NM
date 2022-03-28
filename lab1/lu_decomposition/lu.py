import numpy as np


def lu(a):
    n = a.shape[0]
    l = np.zeros((n, n))
    u = a.copy()
    for i in range(n):
        for j in range(i, n):
            l[j][i] = u[j][i] / u[i][i]

    for k in range (1, n):
        for i in range(k - 1, n):
            for j in range (i, n):
                l[j][i] = u[j][i] / u[i][i]
        for i in range (k, n):
            for j in range(k - 1, n):
                u[i][j] = u[i][j] - l[i][k - 1] * u[k - 1][j]
    return l, u


def solve_lu(a, b, p=True):
    n = a.shape[0]
    l, u = lu(a)
    if p:
        print("Matrix L:\n", l, '\n')
        print("Matrix U:\n", u, '\n')
        print("Matrix L * U:\n", l.dot(u), '\n')
    # L * y = b
    y = []
    for i in range(n):
        s = 0
        for j in range(i):
            s += y[j] * l[i][j]
        y.append(b[i] - s)
    print(y)
    # 2. U * x = y
    x = []
    for i in range(n):
        s = 0
        for j in range(i):
            s += x[j] * u[n - i - 1][n - j - 1]
        x.append((y[n - 1 - i] - s) / u[n - i - 1][n - i - 1])
    x.reverse()
    return np.array(x)


def det(a):
    n = a.shape[0]
    l, u = lu(a)
    det = 1
    for i in range(n):
        det *= u[i][i]
    return det


def inverse(A):
    E = np.eye(A.shape[0])
    inv = []
    for e in E:
        x = solve_lu(A, e, False)
        inv.append(x)
    return np.array(inv).T


def main():
    # input
    np.set_printoptions(precision=2)
    n = int(input("Enter the matrix size: "))
    print(f"Enter matrix A:")
    a = np.array([list(map(float, input().split())) for _ in range(n)])
    print(f"Enter matrix B:")
    b = [float(input()) for _ in range(n)]
    print()

    # LU decomposition
    x = solve_lu(a, b)
    for i, x_i in enumerate(x):
        print(f"x{i + 1} = {x_i}")
    print("x =", x, '\n')

    # Determinant
#    print("Det =", det(a), '\n')
    
    # Inverse
#    print("Inverse:\n", inverse(a))


if __name__ == "__main__":
    main()
