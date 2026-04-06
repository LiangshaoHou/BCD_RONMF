function [U,V]=Jiang_Initial(A,r)
%% Jiang_Initial - SVD-based initialization with sign correction
%
% Purpose:
%   Generate a nonnegative initialization (U, V) for matrix factorization
%   problems with nonnegativity and orthogonality constraints. This method
%   is motivated by the approach proposed in Jiang et al. (2023), where
%   singular vectors are adjusted to align with nonnegative structure before
%   truncation. Compared to NNDSVD, this method is simpler and preserves more
%   of the original SVD geometry while still enforcing nonnegativity.
%
% Inputs:
%   A : (n x m) data matrix
%   r : target factorization rank
%
% Outputs:
%   U : (n x r) nonnegative factor matrix
%   V : (m x r) nonnegative factor matrix (note: not transposed here)
%
% Reference:
%   Jiang, B., et al. (2023).
%   "An exact penalty approach for optimization with nonnegative orthogonality constraints."
%   Mathematical Programming, 198(1), 855–897.
%

%% Step 1: Compute truncated SVD
% Extract leading singular vectors to capture dominant low-rank structure.
% This provides a strong initialization aligned with the principal subspace.
[U, ~, V] = svds(A, r);

%% Step 2: Sign correction for singular vectors
% Singular vectors are defined up to a sign. This step chooses the sign
% so that the majority of the energy lies in the nonnegative part.
% This reduces information loss when truncating negative entries later.
for j = 1:r

    % --- Adjust column j of U ---
    % Identify negative entries
    neg = U(:,j) < 0;

    % Compare energy (Frobenius norm) of negative vs positive parts
    % Flip sign if negative part dominates, so that positive entries carry more mass
    if norm(U(neg,j),'fro') > norm(U(~neg,j),'fro')
        U(:,j) = -U(:,j);
    end

    % --- Adjust column j of V ---
    % Apply the same logic independently to V
    neg = V(:,j) < 0;

    if norm(V(neg,j),'fro') > norm(V(~neg,j),'fro')
        V(:,j) = -V(:,j);
    end
end

%% Step 3: Enforce nonnegativity
% After sign alignment, simply truncate remaining negative entries.
% This produces a feasible starting point for algorithms with
% nonnegativity constraints.
U = max(U, 0);
V = max(V, 0);

%% Optional normalization (commented out)
% These lines suggest a possible column normalization for V,
% which may be useful depending on the downstream algorithm.
% They are left inactive to preserve flexibility in usage.

% X0_norm = sqrt(sum(V.*V));
% V = V ./ X0_norm;
% V = V';