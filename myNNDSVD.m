function [W,H]=myNNDSVD(A,r)
%% myNNDSVD - NNDSVD initialization for Nonnegative Matrix Factorization
%
% Purpose:
%   Compute a structured, nonnegative initialization (W, H) for NMF using
%   the NNDSVD algorithm. This method leverages truncated SVD to extract
%   dominant low-rank structure from the data while enforcing nonnegativity.
%   Compared to random initialization, NNDSVD typically improves convergence
%   speed and solution quality by providing a more informative starting point.
%
% Inputs:
%   A : (n x m) data matrix (assumed nonnegative or approximately so)
%   r : target factorization rank
%
% Outputs:
%   W : (n x r) nonnegative basis matrix
%   H : (r x m) nonnegative coefficient matrix
%
% Reference:
%   Boutsidis, C., & Gallopoulos, E. (2008).
%   "SVD based initialization: A head start for nonnegative matrix factorization."
%   Pattern Recognition, 41(4), 1350–1362.
%

%% Step 1: Compute truncated SVD
% Extract the top-r singular triplets to capture the dominant structure of A.
% This provides the best rank-r approximation in the least-squares sense.
[U, S, V] = svds(A, r);

%% Step 2: Initialize the first component
% The leading singular triplet is guaranteed to have consistent sign structure.
% Taking absolute values ensures nonnegativity while preserving magnitude.
W(:,1) = sqrt(S(1,1)) * abs(U(:,1));
H(1,:) = sqrt(S(1,1)) * abs(V(:,1)');

%% Step 3: Process remaining components
% For each subsequent singular vector pair, NNDSVD decomposes them into
% positive and negative parts and selects the one with larger "energy".
% This avoids cancellation effects and produces meaningful nonnegative factors.
for j = 2:r

    % Extract j-th singular vectors
    x = U(:,j);
    y = V(:,j);

    % Decompose into positive and negative parts
    % This separates opposing directions that would otherwise cancel out.
    xp = max(x, 0);   % positive part of x
    xn = max(-x, 0);  % negative part of x
    yp = max(y, 0);   % positive part of y
    yn = max(-y, 0);  % negative part of y

    % Compute norms to evaluate contribution strength
    xpn = norm(xp); ypn = norm(yp);
    mp = xpn * ypn;   % contribution of positive parts

    xnn = norm(xn); ynn = norm(yn);
    mn = xnn * ynn;   % contribution of negative parts

    % Select the dominant pair (positive or negative)
    % This step ensures that the chosen direction captures maximal variance
    % while remaining nonnegative.
    if mp > mn
        u = xp / xpn;
        v = yp / ypn;
        sigma = mp;
    else
        u = xn / xnn;
        v = yn / ynn;
        sigma = mn;
    end

    % Scale the selected components appropriately using singular values
    % This preserves the magnitude information from SVD while enforcing nonnegativity.
    W(:,j) = sqrt(S(j,j) * sigma) * u;
    H(j,:) = sqrt(S(j,j) * sigma) * v';

end