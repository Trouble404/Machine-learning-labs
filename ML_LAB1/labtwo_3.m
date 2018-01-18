C=[2 -1;-1 2];
X=randn(1000,2);
A=chol(C);
Y=X*A;
N = 50;
plotArray = zeros(N,1);
thRange = linspace(0,2*pi,N);
for n=1:N
 theta = thRange(n);
 u = [sin(theta); cos(theta)]
 yp = Y*u;
 var_empirical = var(yp)
 var_theoretical = u'*C*u
 plotArray(n) =  var_empirical - var_theoretical;
end
plot(plotArray)