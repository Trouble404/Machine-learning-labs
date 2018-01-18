C=[2 1;1 2];X=randn(1000,2);A=chol(C);Y=X*A;% correlated gaussian distribution
N = 10;% test 10,100,1000,10000
plotArray = zeros(N,1);thRange = linspace(0,2*pi,N);% setting coordinate system
for n=1:N
 theta = thRange(n);
 u = [sin(theta); cos(theta)]
 yp = Y*u;
 var_empirical = var(yp)% matlab command 
 var_theoretical = u'*C*u %calculated
 plotArray(n) =  var_empirical - var_theoretical;%difference
end
plot(plotArray)