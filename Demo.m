%% Load and preprocess data
clear
load('ORL_32x32.mat');

% The dataset is transposed so that each column represents a data sample.
% This aligns with the standard matrix factorization form A ≈ U*V,
% where columns of A are approximated by basis vectors in U.
A = fea';

% Retrieve matrix dimensions: n = feature dimension, m = number of samples.
[n,m] = size(A);

% Normalize the data matrix using Frobenius norm.
% This improves numerical stability and ensures that the scale of A
% does not dominate the optimization process.
A = A / norm(A,'fro');  % This normalization stabilizes the experiment

% Set the factorization rank equal to the number of classes.
% This assumes that each class corresponds to one latent component.
r = max(gnd);  % factorization rank


%% Initialization
% The initialization step is critical for nonconvex problems like NMF,
% as it strongly influences convergence speed and solution quality.

% --- Initialization 1: Column sampling-based initialization ---

% Randomly select r columns from A as initial basis vectors.
% This provides a data-driven starting point that preserves structure.
idx = randsample(m, r);
U0 = A(:, idx);

% Normalize each column of U0 to have unit norm.
% This enforces the constraint ||U(:,i)|| = 1 required by RONMF.
for i = 1:length(idx)
    U0(:,i) = U0(:,i) / norm(U0(:,i));
end

% Compute initial coefficients by projecting A onto U0.
% Negative values are truncated to enforce nonnegativity.
tV = max(0, U0' * A);

% For each sample, assign it to the component with the largest coefficient.
% This enforces the "at most one positive entry per column" constraint in V.
[mV, iV] = max(tV);

% Construct sparse matrix V0:
% Each column has only one nonzero entry (cluster assignment),
% making it consistent with the RONMF structure.
V0 = sparse(iV, 1:m, mV, r, m);


%% Alternative initialization methods (optional)
% These methods are commented out but can be used for comparison.

% % Initialization 2: NNDSVD (deterministic and structured initialization)
% % Typically improves convergence by providing a low-rank approximation.
% [U0, V0] = myNNDSVD(A, r);
% 
% % Initialization 3: Custom initialization (e.g., Jiang's method)
% % May incorporate problem-specific heuristics.
% [U0, V0] = Jiang_Initial(A, r);


%% Run BCD-based RONMF algorithm
% The Block Coordinate Descent (BCD) method alternates between updating U and V
% while respecting nonnegativity, unit norm constraints on U,
% and sparsity/assignment constraints on V.

MaxIter = 1000;  % Maximum number of iterations allowed for convergence

% Execute the RONMF solver
% Outputs:
%   F       - objective function values across iterations
%   U, V    - factor matrices such that A ≈ U*V
%   gap_vec - convergence diagnostics (e.g., optimality gap)
[F, U, V, gap_vec] = BCD_RONMF(A, U0, V0, MaxIter);