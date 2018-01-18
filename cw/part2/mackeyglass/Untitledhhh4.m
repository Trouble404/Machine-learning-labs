%predict time series
%run the mackeyglass point first and genreate 2000 samples
clear all;
run('mackeyglass.m')
data=[X T];

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
time=1501:2001;
time=time';
Y=[input_matrix,ones(1481,1)];
 m=iddata([X],[]);
%m=iddata(Y);




n=armax(m,[20 1],'IntegrateNoise',true);  
compare(m,n,1)
%K = 1;
%yp = predict(n,X,K);


%a=yp(1501:2001,1);
%ou=[a,time];
%plot(ou(:,2),ou(:,1));
%legend('Estimation data','Predicted data');