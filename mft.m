%% A theory of meta-factorization: numerical examples
%  M.P.Karpowicz

clear all

% Select test matrix
N       = 5e2;
A       = gallery('randsvd',[3*N,N],1e4);
A       = gallery('cycol',[N,3*N],ceil(N/5));
A       = double(rgb2gray(imread('picopico.jpeg')));

[m,n]   = size(A)
k0      = rank(A)

% Select compression level
k       = ceil(k0*1)

%% Generate basis for column-space and row-space

[Qc,Rc,Pc]  = qr(A);
[Qr,Rr,Pr]  = qr(A');

%% Solutions to linear equations

% Select compression level
k       = k0;

F       = Qc(:,1:k);        B       = randn(m,k);
H       = Qr(:,1:k);        D       = randn(n,k);
[QY,RY] = qr(B'*F,0);       [QX,RX] = qr(H'*D,0);      
Y       = (RY\(QY'*B'))';   X       = (D/RX)*QX';
W       = randn(k,k);
G       = Y'*A*X + W - Y'*F*W*H'*X;
Ar      = F*G*H';

clf 
imagesc(Ar)
axis equal off 
drawnow

err     = norm(Ar-A,'fro')/norm(A,'fro')
err1    = norm(F*Y'*F-F,'fro')
err2    = norm(H'*X*H'-H','fro')

%% CPQR and its mixing matrix

% Select compression level
k       = ceil(k0*.01)

F       = Qc(:,1:k);            B = F;
H       = (Rc(1:k,:)*Pc')';     D = H;
[QY,RY] = qr(B'*F,0);      
[QX,RX] = qr(H'*D,0);      
Y       = (RY\(QY'*B'))';   
X       = (D/RX)*QX';
G       = Y'*A*X
Ar      = F*G*H';

clf 
imagesc(Ar)
axis equal off 
drawnow

err     = norm(Ar-A,'fro')/norm(A,'fro')

%% UTV 

% Select compression level
k       = ceil(k0*.01)

H       = Qr(:,1:k);    D = H;
[QX,RX] = qr(H'*D,0);      
X       = (D/RX)*QX';
G       = A*X;
[Ubar,Sbar,Vbar] = svd(G,'econ');
U       = Ubar;
T       = [Sbar,Ubar'*A*Qr(:,k+1:end)];
V       = Qr*[Vbar,zeros(k,n-k);zeros(k,n-k)',eye(n-k)];
Ar      = U*T*V';

clf 
imagesc(Ar)
axis equal off 
drawnow

err = norm(Ar-A,'fro')/norm(A,'fro')

%% UTV relaxed two-sided SVD-based

% Select compression level
k       = ceil(k0*.01)

F       = Qc(:,1:k);        B       = F;
H       = Qr(:,1:k);        D       = H;
[QY,RY] = qr(B'*F,0);       [QX,RX] = qr(H'*D,0);      
Y       = (RY\(QY'*B'))';   X       = (D/RX)*QX';
G       = Y'*A*X;
[Ubar,Sbar,Vbar] = svd(G);
U       = Y*Ubar;
T       = [Sbar,Ubar'*Y'*A*Qr(:,k+1:end)];
V       = Qr*[Vbar,zeros(k,n-k);zeros(k,n-k)',eye(n-k)];
Ar      = U*T*V';

clf 
imagesc(Ar)
axis equal off 
drawnow

err = norm(Ar-A,'fro')/norm(A,'fro')

%% UTV relaxed two-sided CPQR-based

% Select compression level
k       = ceil(k0*.01)

F       = Qc(:,1:k);      B       = F;
H       = Qr(:,1:k);      D       = H;
[QY,RY] = qr(B'*F,0);     [QX,RX] = qr(H'*D,0);
Y       = (RY\(QY'*B'))'; X       = (D/RX)*QX';
G       = Y'*A*X;
[Qbar,Rbar,Pbar] = qr(G);
U       = Qc(:,1:k)*Qbar;
T       = Rbar;
V       = Qr(:,1:k)*Pbar;
Ar      = U*T*V';

clf 
imagesc(Ar)
axis equal off 
drawnow

err     = norm(Ar-A,'fro')/norm(A,'fro')

%% ULV relaxed two-sided LU-based

% Select compression level
k       = ceil(k0*.01)

F       = Qc(:,1:k);      B       = F;
H       = Qr(:,1:k);      D       = H;
[QY,RY] = qr(B'*F,0);     [QX,RX] = qr(H'*D,0);
Y       = (RY\(QY'*B'))'; X       = (D/RX)*QX';
G       = Y'*A*X;
[Lbar,Ubar,Pbar] = lu(G);
U       = Qc(:,1:k)*Pbar';
T       = Lbar;
V       = Qr(:,1:k)*Ubar';
Ar      = U*T*V';

clf 
imagesc(Ar)
axis equal off 
drawnow

err = norm(Ar-A,'fro')/norm(A,'fro')

%% Generalized Nystrom factorization (by Nakatsukasa)

% Select compression level
k       = ceil(k0*.1)

D       = randn(n,k);   B  = randn(m,ceil(1.5*k)); % Oversampling
F       = A*D;          H  = A'*B;
[Q,R]   = qr(B'*A*D,0);
Ar      = (F/R)*(Q'*H');

clf 
imagesc(Ar)
axis equal off 
drawnow

err     = norm(Ar-A,'fro')/norm(A,'fro')

%% Pseudoinverse formula example

pA      = pinv(A);

[R, cidx] = rref(A);
r       = length(cidx);
B       = A(:,cidx);
D       = R(1:r,:);
err0    = norm(A-B*D,'fro')/norm(A,'fro')

F       = D';           
H       = B;
Ar      = F*((B'*A*D')\H');   

clf 
image(pinv(Ar))
axis equal off 
drawnow

err     = norm(Ar-pA,'fro')/norm(pA,'fro')


%% CUR and Generalized Nystrom factorization

% Select compression level
k       = ceil(k0*.1)

In      = eye(n);       Im      = eye(m);
I       = randperm(m,k);J       = randperm(n,k);
Q       = In(:,J);      P       = Im(:,I);
F       = A*Q;          H       = A'*P;
B       = P;            D       = Q;                % Nystrom method (naive sampling)
B       = F;            D       = H;                % CUR (naive sampling)

warning = [rank(B'*F), rank(H'*D), k]           	% Warning:
[QY,RY] = qr(B'*F,0);   [QX,RX] = qr(H'*D,0);       % If rank(B'*F) = rank(H'*D) = k,
Y       = (RY\(QY'*B'))';   X   = (D/RX)*QX';       %  calculate Y and X.
Y       = (pinv(B'*F)*B')'; X   = D*pinv(H'*D);     % Otherwise, consider using pinv()
G       = Y'*A*X;
Ar      = F*G*H';

clf 
imagesc(Ar)
axis equal off 
drawnow

err     = norm(Ar-A,'fro')/norm(A,'fro')


%% Side note, periodic factorizations (naive implementation)

% Select compression level
k       = ceil(k0*1)

F       = Qc(:,1:k);        B = randn(m,k);
H       = (Rc(1:k,:)*Pc')'; D = H;

% Periodic matix
Z       = diag(roots([1 zeros(1,k-1) -1])); 

% Factorization period
p0      = k;
p       = 3*p0-3
err0    = norm(eye(p0)-Z^p,'fro')

% Periodic factorization
[QY,RY] = qr(B'*F,0);           [QX,RX] = qr(H'*D,0);
Y       = (Z*(RY\(QY'*B')))';   X       = ((D/RX)*QX')*Z;
G       = real(Z^(p-1)*Y'*A*X*Z^(p-1));
Ar      = F*G*H';

clf 
imagesc(Ar)
axis equal off 
drawnow

err     = norm(Ar-A,'fro')/norm(A,'fro')



