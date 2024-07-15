import numpy as np
from forward_sub import *

def forward_gs(A,b,x, niter = 100):
    """
        Forward Gauss Seidel Method - Iteratively solves Ax = b 
            using the following update:
    
            x_{k+1} = x_k + (D + L)^{-1}(b - A * x_k)
    
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
    DL = np.tril(A)         # (D+L)

    for i in range(niter):
        bax = b - A @ x
        res[i] = np.linalg.cond(bax)
        x += forward_sub(DL,bax)          # Gauss - Seidel Iteration

    return x, res