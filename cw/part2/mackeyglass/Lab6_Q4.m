%predict time series
%run the mackeyglass point first and genreate 2000 samples
clear all;
run('mackeyglass.m')
data=[X T];
figure(1),clf
subplot(2,1,1);
plot(T,X);
xlabel('T','FontSize',14);
ylabel('X','FontSize',14);
title('Real mackeyglass prediction','FontSize',14);
tr=data(1:1500,:);
ts=data(1501:2001,:);
p=20;
y=data(21:1501,1);
input_matrix=zeros(1481,20);
for i=1:1481
    for j=1:20
   input_matrix(i,j)=tr(j+i-1,1);
    end
end
%%Best linear predictor
Y=[input_matrix,ones(1481,1)];
w=inv(Y'*Y)*Y'*y;
fh=Y*w;
%check on the test data
for j=1:481
    input_matrix1=ts(j:j+19,1);
    Y_ts=[input_matrix1' ones(1,1)];
    ts(20+j,1)=Y_ts*w;
end  
subplot(2,1,2);
pred1=[tr;ts];
plot(pred1(:,2),pred1(:,1));
xlabel('T','FontSize',14);
ylabel('X','FontSize',14);
title('Best Linear Predictor','FontSize',14);
%neural network
 [net]=feedforwardnet(20);
 [net]=train(net,input_matrix',y');
 
 for j=1:481
     input_matrix2=ts(j:j+19,1);
     ts(20+j,1)=net(input_matrix2);
 end  
 figure(2),clf
 subplot(2,1,1);
 plot(T,X);
 xlabel('T','FontSize',14);
 ylabel('X','FontSize',14);
 title('Real mackeyglass prediction','FontSize',14);
 subplot(2,1,2);
 pred2=[tr;ts];
 plot(pred2(:,2),pred2(:,1));
 xlabel('T','FontSize',14);
 ylabel('X','FontSize',14);
 title('Iterated network prediction','FontSize',14);



   
    


