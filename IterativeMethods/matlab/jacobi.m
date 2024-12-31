function [x, res] = jacobi(A, b, x, niter)
%   Jacobi Method - Iteratively solves Ax = b using the following update:
%
%           x_{k+1} = x_k + D^{-1}(b - A * x_k)
%
%   Inputs: A - matrix in lnear equation
%           b - known vector in linear equation
%           x - initial guess to linear equation
%           niter - number of iterations 
%
%   Outputs: x - solution to linear equation
%            res - norm of resididual vector

res = zeros(1,niter); 
Dinv = 1 ./ diag(A);            % (D^{-1})

for i = 1:niter

    res(i) = norm(b - A * x);
    x = x + Dinv .* (b - A * x);     % Jacobi iteration     
end