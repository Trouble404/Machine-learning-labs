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
contour(xRange, yRange, P2, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
plot(m2(1),m2(2), 'r*', 'LineWidth', 4);