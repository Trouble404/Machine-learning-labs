C=[2 1;1 2];
X=randn(1000,2);
A=chol(C);
Y=X*A;
for theta=0:0.002:5;

 u = [sin(theta); cos(theta)]
 yp = Y*u;
var_empirical = var(yp)
var_theoretical = u'*C*u;

plot(theta,var_theoretical,'b.');
plot(theta,var_empirical,'c.');
hold on;
end