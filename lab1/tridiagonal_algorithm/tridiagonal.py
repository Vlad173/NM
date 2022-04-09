import numpy as np
from tabulate import tabulate


def tdma(a, b, c, d):
    n = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, n):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1] 
        dc[it] = dc[it] - mc * dc[it - 1]
        	    
    x = np.copy(b)
    x[-1] = dc[-1] / bc[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x


def get_diagonals(m):
    n = len(m)
    a, b, c = [], [], [] 
    
    for i in range(n):
        if i != 0:
            a.append(m[i][i - 1])
        if i != n - 1:
            c.append(m[i][i + 1])
        b.append(m[i][i])
    return a, b, c


def main():
    # input
    np.set_printoptions(precision=5)
    n = int(input())
    a = np.array([list(map(float, input().split())) for _ in range(n)])
    x, y, z = get_diagonals(a)
    b = np.array([float(input()) for _ in range(n)])
    print("Matrix A:\n", tabulate(a), '\n')
    print("Matrix B:")
    print(b, '\n')

    print("TDMA:")
    x = tdma(x, y, z, b)
    print('x =', x, '\n')

    print("Numpy:")
    print('x =', np.linalg.solve(a, b))


if __name__ == "__main__":
    main()
