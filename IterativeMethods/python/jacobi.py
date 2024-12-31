import numpy as np

def jacobi(A, b, x, niter = 100):
    """
        Jacobi Method - Iteratively solves Ax = b using the following update:

            x_{k+1} = x_k + D^{-1}(b - A * x_k)

        Inputs: A - matrix in lnear equation
            b - known vector in linear equation
            x - initial guess to linear equation
            niter - number of iterations 

        Outputs: x - solution to linear equation
            res - norm of resididual vector
    """   
    res = np.zeros(niter)
    Dinv = 1.0/ np.diag(A)

    for i in range(niter):
        res[i] = np.linalg.norm(b - A @ x)
        x += Dinv * (b - A @ x)             # Jacobi Iteration

    return x, res
