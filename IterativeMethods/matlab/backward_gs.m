function [x, res] = backward_gs(A, b, x, niter)
%   Backward Gauss Seidel Method - Iteratively solves Ax = b 
%           using the following update:
%
%           x_{k+1} = x_k + (D + U)^{-1}(b-Ax_k)
%
%   Inputs: A - matrix in lnear equation
%           b - known vector in linear equation
%           x - initial guess to linear equation
%           niter - number of iterations 
%
%   Outputs: x - solution to linear equation
%            res - norm of resididual vector

res = zeros(1, niter);
DU = triu(A);            % (D + U)

for i = 1:niter
    
    res(i) = norm(b - A * x);
    x = x + DU \ (b - A * x);        % Gauss Seidel iteration  
    
end