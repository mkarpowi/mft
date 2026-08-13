% NETWORK_EXAMPLE  The worked example of Section 4 of "A theory of
% meta-factorization": the replacement criterion on a resistor network.
%
% Everything below the header is byte-for-byte the listing printed in the
% paper.  Running it prints the two output blocks displayed there, in order:
% rows 1-2 are the first block, rows 3-4 the second.  It is self-contained --
% no toolbox, no other file, no random stream.
%
%   Expected output:
%
%     rank 4   test 3.114894e-14   err 3.752010e-14   det  1.000
%     rank 3   test 3.291403e+00   err 3.415650e+00   det -0.000
%     rank 5   test 3.044879e-14   err 4.435485e-14   det  1.000
%     rank 4   test 6.895672e-15   err 1.815507e-14   det  0.000
%
% Every core is built with PINV, as printed.  Building the augmented (k = m)
% core instead with MATLAB's left/right division is algebraically equivalent
% but changes the roundoff digits of the residuals, so the two must not be
% mixed.
%
% The refused residual is sqrt(390)/6 exactly; the accepted interface between
% two spanning trees is unimodular, det = 1, because edge-vector matrices are
% totally unimodular.

%% the network, and the Laplacian it defines
n = 5;
E = [1 2; 2 3; 1 3; 3 4; 4 5; 3 5];   g = [2 1 3 1 4 2]';
B = inc(E,n);   A = B'*diag(g)*B;            % rank r = n-1 = 4

%% at k = r: F = H = the edge vectors of a spanning tree
T1 = [1 2; 2 3; 3 4; 4 5];   T2 = [1 3; 2 3; 3 5; 4 5];
D  = [1 2; 2 3; 1 3; 4 5];               % triangle + lone edge: disconnected
F  = inc(T1,n)';   H = F;    G  = pinv(F)*A*pinv(F)';

%% at k = m: adjoin the constant vector, the null direction
Fm = [F ones(n,1)];          Gm = pinv(Fm)*A*pinv(Fm)';

%% the criterion, on four candidates
cand = {inc(T2,n)', inc(D,n)', [inc(T2,n)' ones(n,1)], [F zeros(n,1)]};
base = {F, F, Fm, Fm};       core = {G, G, Gm, Gm};
for c = 1:4
    Fc = cand{c};   Yc = pinv(Fc);           % a {1}-inverse is admissible
    Sc = Yc*base{c};   Ec = base{c} - Fc*Sc; % interface and mismatch
    test = norm(Ec*core{c},'fro');           % the criterion
    err  = norm(A - Fc*(Sc*core{c})*base{c}','fro');
    fprintf('rank %d   test %11.6e   err %11.6e   det %6.3f\n', ...
            rank(Fc), test, err, det(Sc));
end

function B = inc(E,n)                        % incidence matrix of an edge list
    B = zeros(size(E,1),n);
    for e = 1:size(E,1),  B(e,E(e,1)) = 1;  B(e,E(e,2)) = -1;  end
end
