C=[2 1;1 2];
X=randn(100,2);
m1 = [0 2];
X1 = X + kron(ones(100,1),m1);
A=chol(C);
Y=X1*A;
plot(Y(:,1),Y(:,2),'mx');
hold on;

C=[2 1;1 2];
X=randn(100,2);
m1 = [1.5 0];
X1 = X + kron(ones(100,1),m1);
A=chol(C);
Y=X1*A;
plot(Y(:,1),Y(:,2),'bx');