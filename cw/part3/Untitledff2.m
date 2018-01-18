clear all;
data1=xlsread('close3.xlsx');
data2=xlsread('vol.xlsx');
time = 1:1265;
time = time'
data3 = [data1 , data2];
close_data=[data1,time];
vol_data=[data2,time];

for i=1:1246
  input_matrix(i,:)=close_data(i:i+19);
end
Y1=[input_matrix ones(1246,1)];
y1=close_data(20:1265,1);

for i=1:1246
  input_matrix1(i,:)=vol_data(i:i+19);
end
Y2=[input_matrix1 ones(1246,1)];
y2=close_data(20:1265,1);

Y3=[input_matrix,input_matrix1];
y3=[y1,y2];

[net]=feedforwardnet(20);
net.trainFcn = 'trainbr';
[net]=train(net,Y3',y3');
[output1]=net(Y3');
err_nn=immse(output1',y3);

ou2=zeros(2,1246);
f_in1=data1(1:20);
f_in2=data2(1:20);
f_in=[f_in1;f_in2];

for  j=1:1246
    b=net(f_in);
    ou2(1,j)=b(1,1);
    ou2(2,j)=b(2,1);
    f1=[f_in(2:20);b(1,1)];
    f2=[f_in(22:40);b(2,1)];
    f_in=[f1;f2];
end 
time3=20:1265;
out1=[ou2',time3'];