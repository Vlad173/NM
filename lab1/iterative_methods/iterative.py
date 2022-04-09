import numpy as np
import math
import copy
from tabulate import tabulate

# Погрешность
# Euclidean distance
def accuracy(x_old, x_new, eps):
    sum_up = 0
    sum_low = 0
    for k in range(len(x_old)):
        sum_up += (x_new[k] - x_old[k]) ** 2
        sum_low += (x_new[k]) ** 2
        
    return math.sqrt(sum_up / sum_low) < eps


def iterative(a, b, eps=0.001):
    count = len(b)
    x = np.array([0. for k in range(count)])
    
    it = 0
    print("Iterative method:")
    while it < 100:
        x_prev = copy.deepcopy(x)
        for i in range(count):
            s = 0
            for j in range(count):
                if j != i:
                    s = s + a[i][j] * x_prev[j] 
            x[i] = b[i] / a[i][i] - s / a[i][i]
        print(f'{it + 1}:', x)
        if accuracy(x_prev, x, eps):
            break
        it += 1
    return x    
            

def seidel(a, b, eps=0.001):
    count = len(b)
    x = np.array([0. for k in range(count)])
    
    it = 0
    print("Gauss-Seidel method:")
    while it < 100:
        x_prev = copy.deepcopy(x)
        for i in range(count):
            s = 0
            for j in range(count):
                if j < i:
                    s = s + a[i][j] * x[j] 
                elif j > i:
                    s = s + a[i][j] * x_prev[j] 
            x[i] = b[i] / a[i][i] - s / a[i][i]
        print(f'{it + 1}:', x)
        if accuracy(x_prev, x, eps):
            break
        it += 1
    return x    


def main():
    # input
    np.set_printoptions(precision=8)
    n = int(input())
    eps = float(input())
    a = np.array([list(map(float, input().split())) for _ in range(n)])
    b = np.array([float(input()) for _ in range(n)])
    print("Matrix A:\n", tabulate(a), '\n')
    print("Matrix B:\n", b, '\n')

    x = iterative(a, b, eps)
    print("x =", x)
    print()
    x = seidel(a, b, eps) 
    print("x =", x)

    print("\nNumpy:")
    print("x =", np.linalg.solve(a, b))


if __name__ == "__main__":
    main()
