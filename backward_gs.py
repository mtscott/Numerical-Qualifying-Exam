import numpy as np
from back_sub import *

def backward_gs(A,b,x, niter = 100):
    """
        Backward Gauss Seidel Method - Iteratively solves Ax = b 
            using the following update:
    
            x_{k+1} = x_k + (D + U)^{-1}(b - A * x_k)
    
        Inputs:
            A - matrix in lnear equation
            b - known vector in linear equation
            x - initial guess to linear equation
            niter - number of iterations 
    
       Outputs: 
            x - solution to linear equation
            res - norm of resididual vector
    """
    res = np.zeros(niter)
    DU = np.triu(A)         # (D+U)

    for i in range(niter):
        bax = b - A @ x
        res[i] = np.linalg.cond(bax)
        x += back_sub(DU,bax)          # Gauss - Seidel Iteration

    return x, res