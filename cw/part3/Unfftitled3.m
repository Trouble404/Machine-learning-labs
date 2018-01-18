clear all;
data1=xlsread('vol.xlsx');

time = 1:1265;
time = time'
close_data = [data1 , time];

for i=1:1246
  input_matrix(i,:)=close_data(i:i+19);
end
Y=[input_matrix ones(1246,1)];
y=close_data(20:1265,1);

%%nn

[Y1,PS1] = mapminmax(Y);
[y1,PS2] = mapminmax(y);
[net]=feedforwardnet(20);
net.trainFcn = 'trainbr';
[net]=train(net,Y1',y1');
[output1]=net(Y1');
err_nn=immse(output1',y1);

%for  j=1:1246
   % input_matrix1=data1(j:j+19,1);
    %Y1_ts=[input_matrix1' ones(1,1)];
    %ou(j)=net(Y1_ts');
%end

%time1=20:1265;
%time1=time1'
%out=[ou',time1];

ou2=zeros(1,1246);
f_in=data1(1:20);
for  j=1:1246
    input_matrix2=f_in;
    Y2_ts=[input_matrix2' ones(1,1)];
    ou2(j)=net(Y2_ts');
    f_in=[f_in(2:end);ou2(j)];
end 
mapminmax('reverse',ou2,PS1)  
time2=1:1246;
time2=time2'
out1=[ou2',time2];
plot(out1(:,2),out1(:,1));