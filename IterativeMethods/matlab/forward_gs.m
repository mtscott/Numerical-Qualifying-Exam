function [x, res] = forward_gs(A, b, x, niter)
%   Forward Gauss Seidel Method - Iteratively solves Ax = b 
%           using the following update:
%
%           x_{k+1} = x_k + (D + L)^{-1}(b - A * x_k)
%
%   Inputs: A - matrix in lnear equation
%           b - known vector in linear equation
%           x - initial guess to linear equation
%           niter - number of iterations 
%
%   Outputs: x - solution to linear equation
%            res - norm of resididual vector

res = zeros(1,niter); 
DL = tril(A);            % (D+L)

for i = 1:niter
    
    res(i) = norm(b - A * x);
    x = x + DL\(b-A*x);          % Gauss Seidel iteration
    
end
