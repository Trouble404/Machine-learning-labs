clear all;
run('mackeyglass.m')
Ntr=1500;
Nts=500;
data=[X T];
Tr=data(1:1500,:);
Ts=data(1501:2001,:);
Ts2=data(1501:2001,1);
Ts1=Ts;
Ts3=Ts;
p=20;
num1=1;
for i=1:(Ntr-p+1)
  input_matrix(i,:)=X(num1:num1+19);
  num1=num1+1;
end
num2=1501;
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
%err_linear=immse(fh,Xh);
figure(1),clf
plot(data(1521:2001,2),data(1521:2001,1));
hold on;
plot(Th,fh);
xlabel('T','FontSize',14);
ylabel('X','FontSize',14);
title('Best Linear Predictor','FontSize',14);
legend('true mackeyglass line','our predict line');
%%nn
[net]=feedforwardnet(20);
[net]=train(net,Y',y');
[output]=net(Y1');
[output1]=net(Y');
err_nn=immse(output',Xh);
%plot
figure(2),clf
plot(data(1521:2001,2),data(1521:2001,1));
hold on;
plot(Th,output);
xlabel('T','FontSize',14);
ylabel('X','FontSize',14);
title('Iterated Network Prediction','FontSize',14);
legend('true mackeyglass line','our predict line');
%free running code
for j=1:481
    input_matrix2=Ts(j:j+19,1);
    Y_ts=[input_matrix2' ones(1,1)];
    Ts(20+j,1)=Y_ts*w;
end  
for  j=1:481
    input_matrix3=Ts1(j:j+19,1);
    Y1_ts=[input_matrix3' ones(1,1)];
    Ts1(20+j,1)=net(Y1_ts');
end  

for  j=1:1481
    input_matrix4=Ts3(j:j+19,1);
    Y2_ts=[input_matrix4' ones(1,1)];
    Ts3(20+j,1)=net(Y2_ts');
end 
B=Ts3(:,1);
time=1501:3001;
time=time'
A=[B,time];
err_freelinear=immse(Ts(:,1),Ts2);
err_freenn=immse(Ts1(:,1),Ts2);
figure(3),clf
plot(data(1521:2001,2),data(1521:2001,1));
hold on;
plot(Ts(:,2),Ts(:,1));
xlabel('T','FontSize',14);
ylabel('X','FontSize',14);
title('Best Linear Predictor in Free Running Code','FontSize',14);
legend('true mackeyglass line','our predict line');
figure(4),clf
plot(T,X);
hold on;
plot(Ts1(:,2),Ts1(:,1));
xlabel('T','FontSize',14);
ylabel('X','FontSize',14);
title('Iterated Network Prediction in Free Running Code','FontSize',14);
legend('true mackeyglass line','our predict line');
