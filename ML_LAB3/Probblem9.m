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
% Nearest neighbour classifier
% (Caution: The follwing code is very inefficcient --why?)
X = [X1;X2];
N1 = size(X1,10);
N2 = size(X2,1);
y = [ones(N1,1); -1*ones(N2,1)];
d = zeros(N1+N2-1,1);
nCorrect = 0;
for jtst = 1:(N1+N2)
    %pick a point to test
    %
    xtst = X(jtst,:);
    ytst = y(jtst);
    
    %All others form the tranining set
    %
    jtr = setdiff(1:N1+N2,jtst);
    Xtr = X(jtr,:);
    ytr = y(jtr,1);
    
    %Compute all distances from test to training points
    %
    for i=1:(N1+N2-1)
        d(i) = norm(Xtr(i,:)-xtst);
    end
    
    %which one is the closest?
    %
    [imin] = find(d == min(d));
    
    %Does the nearest point have the same class label?
    %
    if ( ytr(imin(1)) * ytst > 0 )
        nCorrect = nCorrect + 1;
    else
        disp('Incorrect classification');
    end
end

%Percentage correst
%
pCorrect = nCorrect*100/(N1+N2);
disp(['Nearest neighbour accuracy:  ' num2str(pCorrect)]);