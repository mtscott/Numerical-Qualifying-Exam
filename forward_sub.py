import numpy as np

def forward_sub(L, b):
    """
        x = forward_sub(L, b) is the solution to L x = b
        L must be a lower-triangular matrix
        b must be a vector of the same leading dimension as L
    """
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n):
        tmp = b[i]
        for j in range(i-1):
            tmp -= L[i,j] * x[j]
        x[i] = tmp / L[i,i]
    return x
