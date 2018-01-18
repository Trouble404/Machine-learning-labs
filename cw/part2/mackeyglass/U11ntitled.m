init_sys = idpoly([1 -0.99],[],[1 -1 0.2]);
opt = simOptions('AddNoise',true);
u = iddata([],zeros(400,0),1);
data = sim(init_sys,u,opt);

na = 1;
nb = 2;
sys = armax(data(1:200),[na nb]);

K = 4;
yp = predict(sys,data,K);

plot(data(201:400),yp(201:400));
legend('Estimation data','Predicted data');