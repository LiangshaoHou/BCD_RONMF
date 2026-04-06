function [F,U,V,gap_vec]=BCD_RONMF(A,U0,V0,MaxIter)
% BCD_RONMF: Block Coordinate Descent for RONMF (Rescaled Orthogonal NMF)
%
% This function solves the following optimization problem:
%   min_{U,V} ||A - U V||_F
%   s.t.      U >= 0,  ||U(:,i)||_2 = 1  (column-wise normalization)
%             V >= 0,  each column of V has at most one positive entry
%
% INPUT:
%   A        : (n x m) nonnegative data matrix
%   U0, V0   : initial factors (feasible or approximately feasible)
%   MaxIter  : maximum number of iterations
%
% OUTPUT:
%   F        : vector of objective values 
%   U, V     : factor matrices
%   gap_vec  : optimality gap sequence (used as stopping criterion)
%
% METHOD:
%   Alternating Block Coordinate Descent (BCD):
%     - Fix V, update U via its explict-solution
%     - Fix U, update V via its explict-solution
% Please refer to our paper for more details:  
% **L. Hou, D. Chu, and L.-Z. Liao.**  
% *A fast block coordinate descent method for orthogonal nonnegative matrix factorization.*  
% *(To appear in SIAM Journal on Matrix Analysis and Applications, SIMAX).*

% Initialization
F = [];                              % Objective value history
[n,m] = size(A);                     % Dimensions of input matrix
r = size(U0,2);                      % Target rank
normA2 = sum(A(:).^2);               % Squared Frobenius norm of A
Xind = 1:m;                          % Column indices for constructing V

U = U0; 
V = V0;
gap_vec = [];                        % Store optimality gaps

% Main iteration loop
for i = 1:MaxIter
    
    %% ===== Update U (fix V) =====
    % Compute gradient-like term A*V'
    AV = A * V';
    
    % Enforce nonnegativity
    tU = max(0, AV);
    
    % Normalize each column to unit norm
    nU = sqrt(sum(tU.^2));           % Column-wise ℓ2 norms
    
    % Handle zero columns (avoid division by zero)
    Zind = find(nU == 0);
    if ~isempty(Zind)
        nU(Zind) = 1;                % Prevent division by zero
        
        % For zero columns, assign a unit vector at max location
        [~, mind] = max(AV(:, Zind));
        tU(sub2ind([n,r], mind, Zind)) = 1;
    end
    
    % Final normalized update
    U = tU ./ nU;
    
    %% ===== Update V (fix U) =====
    % Compute projection U'*A
    tV = max(0, U' * A);             % Enforce nonnegativity
    
    % For each column, keep only the largest entry (1-sparse constraint)
    [mV, iV] = max(tV);              % mV: values, iV: indices
    
    % Construct sparse matrix V with one nonzero per column
    V = sparse(iV, Xind, mV, r, m);
    
    %% ===== Compute stopping criteria =====
    % Optimality gap (problem-dependent measure)
    [gap1, Z1] = OptGap1(A, U, V);
    gap_vec = [gap_vec, gap1];
    
    % Objective function:
    %   ||A - UV||_F / ||A||_F
    % Using equivalent fast computation:
    F = [F, sqrt(1 - sum(mV.^2) / normA2)];
    
    %% ===== Convergence check =====
    % Stop if both objective and optimality gap stabilize
    if i >= 2 && ...
       abs(gap_vec(end) - gap_vec(end-1)) < 1e-12 && ...
       abs(F(end) - F(end-1)) < 1e-10
        break;
    end
end
