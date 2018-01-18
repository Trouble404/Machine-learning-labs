
m1 = [0 3]';
m2 = [2 1]';
C1 = [2 1 ; 1 2];
C2 = [1 0; 0 1];
wF = inv(C1+C2)*(m1-m2);
xx = -6:0.1:6;
yy = xx*wF(2)/wF(1);
plot(xx,yy,'r', 'LineWidth',2);