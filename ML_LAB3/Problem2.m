numGrid = 100;
m1 = [0 3]';
m2 = [2 1]';
C1 = [2 1 ; 1 2];
C2 = [1 0; 0 1];
xRange = linspace(-4, 6.0, numGrid);
yRange = linspace(-4, 6.0, numGrid);
P1 = zeros(numGrid, numGrid);
P2 = P1;
for i=1:numGrid
    for j=1:numGrid
        x = [yRange(j) xRange(i)]';
        P1(i,j) = mvnpdf(x',m1',C1);
        P2(i,j) = mvnpdf(x',m2',C2);
    end
end
Pmax = max(max([P1 P2]));
figure(1), clf,
contour(xRange, yRange, P1, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
hold on
plot(m1(1),m1(2), 'b*', 'LineWidth', 4);
hold on
contour(xRange, yRange, P2, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
hold on
plot(m2(1),m2(2), 'r*', 'LineWidth', 4);
hold on
N=200;
X1 = mvnrnd(m1,C1,N);
X2 = mvnrnd(m2,C2,N);
plot(X1(:,1),X1(:,2),'bx',X2(:,1),X2(:,2),'ro');
grid on

a=(C1)^-1 - (C2)^-1;
w= 2*(m1'*((C1)^-1) - m2'*((C2)^-1));
b = m1'*inv(C1)*m1-m2'*inv(C2)*m2 + log(det(C1)/det(C2))/log(exp(1));

syms x y
f= [x y]*a*[x;y]-w*[x;y]+b;
ezplot(f)
delete(get(gca,'title'));
hold on;