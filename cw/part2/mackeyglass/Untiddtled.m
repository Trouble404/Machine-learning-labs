%predict time series
%run the mackeyglass point first and genreate 2000 samples
clear all;
run('mackeyglass.m')
data=[X T];
data1=data;
figure(1),clf
subplot(2,1,1);
plot(T,X);
xlabel('T','FontSize',14);
ylabel('X','FontSize',14);
title('Real mackeyglass prediction','FontSize',14);
tr=data(1:1500,:);
tr1=tr;
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
for j=1:501
    input_matrix1=data(1481+j:1481+j+19,1);
    Y_ts=[input_matrix1' ones(1,1)];
    input=Y_ts*w;
    time=1500+j;
    output=[input,time];
    tr1=[tr1;output];
end  
error=data(:,1)-tr1(:,1);
subplot(2,1,2);
pred1=[tr;ts];
plot(data(:,2),error);
xlabel('T','FontSize',14);
ylabel('X','FontSize',14);
title('Best Linear Predictor','FontSize',14);