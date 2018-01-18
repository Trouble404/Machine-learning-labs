clear all;
run('mackeyglass.m')
Ntr=1500;
Nts=500;
data=[X T];
Tr=data(1:1500,:);
Ts=data(1501:2001,:);
Ts2=data(1501:2001,1);
Ts1=Ts;
p=20;
num1=1;
for i=1:(Ntr-p+1)
  input_matrix(i,:)=X(num1:num1+19);
  num1=num1+1;
end
num2=1500;
for j=1:481
    input_matrix1(j,:)=X(num2:num2+19,:);
    num2=num2+1;
end
y=data(21:1501,1);
yts=Ts(21:501,1);
Y=[input_matrix ones(1481,1)];
w=inv(Y'*Y)*Y'*y;
Y1=[input_matrix1 ones(481,1)];
fh=Y1*w;
f=Y*w;
Th=T(1521:2001,1);
Xh=X(1521:2001,1);
err_linear=immse(fh,Xh);
figure(1),clf
plot(data(1521:2001,2),data(1521:2001,1));
hold on;
plot(Th,fh);
xlabel('T','FontSize',14);
ylabel('X','FontSize',14);
title('Best Linear Predictor','FontSize',14);
legend('true mackeyglass line','our predict line');