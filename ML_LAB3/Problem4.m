numGrid = 50;
m1 = [0 2]';
m2 = [1.7 2.5]';
C1 = [2 1 ; 1 2];
C2 = C1;
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

wF = inv(C1+C2)*(m1-m2);
xx = -6:0.1:6;
yy = xx*wF(2)/wF(1);
plot(xx,yy,'r', 'LineWidth',2);
hold on;
p1 = X1*wF;
p2 = X2*wF;

plo = min([p1; p2]);
phi = max([p1; p2]);
[nn1,xx1] = hist(p1);
[nn2,xx2] = hist(p2);
hhi = max([nn1 nn2]);
figure(2),
subplot(211),bar(xx1,nn1);
axis([plo phi 0 hhi]);
title('Distribution of Projections' , 'FontSize',16)
ylabel('Class 1','FontSize',14)
subplot(212),bar(xx2,nn2);
axis([plo phi 0 hhi])
ylabel('Class 2','FontSize',14)
