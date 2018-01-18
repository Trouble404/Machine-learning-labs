C=[2 -1;-1 2]; % covariance matrix
X=randn(1000,2);% generate 1000*2 matrix with Gassian distribution
A=chol(C); % A^t&A = C
Y=X*A;
% mark Gaussian distribution with cyan potins
% mark another Y with magenta x
plot(X(:,1),X(:,2),'c.', Y(:,1),Y(:,2),'mx');