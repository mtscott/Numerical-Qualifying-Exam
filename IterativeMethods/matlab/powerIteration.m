function [eval, evec] = powerIteration(A, rank, niter)
%   Power Iteration Method - Iteratively solves Ax = lambda x 
%           using the following update:
%
%           x_{k+1} = Ax_k/ ||Ax_k||
%
%   Inputs: A - matrix in linear equation
%           rank - number of eigenvectors to find (default 1)
%           niter - number of iterations 
%
%   Outputs: eval - solution to dominant eigenvalue (computed using Rayleigh Quotient)
%            evec - solution to dominant eigenvector (computed using 

    n = size(A,1);
    v = rand(n,1);
    Av = A * v;
    Av = Av/norm(Av);
    for i = 1 : 50-1
        Av = A * Av;
        normAv = norm(Av);
        Av = Av/normAv;
    end

    evec = Av;
    eval = Av' * (A* Av) / (Av' * Av);
end
