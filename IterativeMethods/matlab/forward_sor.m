function [x, res] = forward_sor(A, b, x, w, niter)
%   Forward Gauss Seidel Method - Iteratively solves Ax = b 
%           using the following update:
%
%           x_{k+1} = x_k + (D + wL)^{-1}(wb - [wU + (w - 1)D] * x_k)
%
%   Inputs: A - matrix in lnear equation
%           b - known vector in linear equation
%           x - initial guess to linear equation
%           w - relaxation parameter
%           niter - number of iterations 
%
%   Outputs: x - solution to linear equation
%            res - norm of resididual vector

res = zeros(1,niter); 
DwL = diag(diag(A)) + w * tril(A,-1);      % D + w L
UwD = -w * triu(A,  1) + (1.0 - w) * diag(diag(A));          % U + w^{-1}D



for i = 1:niter
    
    res(i) = norm(b - A * x);
    x = DwL \ (UwD * x) - w*b; % SOR iteration
    
end
