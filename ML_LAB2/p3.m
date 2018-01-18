C= [2 1;1 2];
A = chol(C)
X = randn(100,2)
m1 =[0 2];
m2 =[1.5 0]
X1 = X+kron(ones(100,1),m1);
Y1 = X1*A;
X2 = X+kron(ones(100,1),m2);
Y2 = X2*A
plot(Y1(:,1),Y1(:,2),'mx');
hold on;
plot(Y2(:,1),Y2(:,2),'bx');
hold on;
m3 = [0;2];
m4 = [1.5;0];
c = inv(C);
w = 2*c*(m4-m3);
b = (m1*c*m3-m2*c*m4);
x = linspace(-8,8);
y = (-w(1)/w(2))*x -(b/w(2)); % y=wt*x+b
plot(x,y,'k');

