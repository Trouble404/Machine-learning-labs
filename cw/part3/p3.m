clear all;
data1 = csvread('close.csv',1)
time = 1:1259;
time = time'
close_data = [data1 , time];


for i=1:1240
  input_matrix(i,:)=close_data(i:i+19);
end
Y=[input_matrix ones(1240,1)];
y=close_data(20:1259,1);

%%nn
%[y1,PS] = mapminmax(Y)
[net]=feedforwardnet(20);
net.trainFcn = 'trainbr';
[net]=train(net,Y',y');
[output1]=net(Y');
err_nn=immse(output1',y);

for  j=1:1240
    input_matrix1=data1(j:j+19,1);
    Y1_ts=[input_matrix1' ones(1,1)];
    ou(j)=net(Y1_ts');
end  
 
time1=20:1259;
time1=time1'
out=[ou',time1];

ou2=zeros(1,1240);
f_in=data1(1:20);
for  j=1:1240
    input_matrix2=f_in;
    Y2_ts=[input_matrix2' ones(1,1)];
    ou2(j)=net(Y2_ts');
    f_in=[f_in(2:end);ou2(j)];
end 
out1=[ou2',time1];