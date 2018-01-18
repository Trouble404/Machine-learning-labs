clear all
run('mackeyglass.m')
x=X;
% 该脚本用来做NAR神经网络预测
lag=20;    % 自回归阶数
iinput=x;    % x为原始序列（行向量）
n=length(iinput);

%准备输入和输出数据
inputs=zeros(lag,n-lag);
for i=1:n-lag
    inputs(:,i)=iinput(i:i+lag-1)';
end
input1=inputs';
input2=input1(1:1481,:);
input3=input2;
targets=x(1:1481);

%线性回归
data=[X T];
tr1=data(1:1500,:);
ts=data(1501:2001,:);
p=20;
y=data(21:1501,1);
input_matrix=zeros(1481,20);
for i=1:1481
    for j=1:20
   input_matrix(i,j)=tr1(j+i-1,1);
    end
end
%%Best linear predictor
Y=[input_matrix,ones(1481,1)];
w=inv(Y'*Y)*Y'*y;
fh=Y*w;

f_out1=zeros(1,481);
f_out1=f_out1';%预测输出
% 多步预测时，用下面的循环将网络输出重新输入
for i=1:481
    f_in1=input3(1481-1+i: 1500-1+i);
    term1=[f_in1 ones(1,1)];
    f_out1(i)=term1*w;
    f_in1=[f_in1(2:end),f_out1(i)];
end
% 画出预测图
ou1 = f_out1;
time4=[1501:1981];
time5=time4';
out1 =[ou1,time5];
plot(out1(:,2),out1(:,1),'r');
hold on;
plot(out1(:,2),X(1501:1981),'b');

%创建网络
hiddenLayerSize = 20; %隐藏层神经元个数
net = fitnet(hiddenLayerSize);

% 避免过拟合，划分训练，测试和验证数据的比例
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%训练网络
[net,tr] = train(net,input2',targets');
%% 根据图表判断拟合好坏
yn=net(input2');
errors=targets-yn;
figure, ploterrcorr(errors)                      %绘制误差的自相关情况（20lags）
%figure, parcorr(errors)                          %绘制偏相关情况
%[h,pValue,stat,cValue]= lbqtest(errors)         %Ljung－Box Q检验（20lags）
%figure,plotresponse(con2seq(targets),con2seq(yn))   %看预测的趋势与原趋势
%figure, ploterrhist(errors)                      %误差直方图
figure, plotperform(tr)                          %误差下降线

%% 下面预测往后预测几个时间段
fn=500;  %预测步数为fn

f_in=X(n-fn-lag: n-fn-1)';
kk=f_in;
f_out=zeros(1,fn);  %预测输出
% 多步预测时，用下面的循环将网络输出重新输入
for i=1:fn
    f_out(i)=net(f_in');
    term = f_out(i);
    f_in=[f_in(2:end),f_out(i)];
end
% 画出预测图
ou = f_out';
time=[1501:1500+fn];
time1=time';
out =[ou,time1];
plot(out(:,2),out(:,1),'r');
hold on;
plot(out(:,2),X(1501:1500+fn),'b');
%figure,plot(1:1501,iinput,'b',1501:2001,[iinput(end),f_out],'r')

%% 下面预测往后预测几个时间段
fn1=1000;  %预测步数为fn

f_in2=X(n-fn1-lag: n-fn1-1)';
f_out2=zeros(1,fn1);  %预测输出
% 多步预测时，用下面的循环将网络输出重新输入
for i=1:fn1
    f_out2(i)=net(f_in2');
    term = f_out2(i);
    f_in2=[f_in2(2:end),f_out2(i)];
end
% 画出预测图
ou = f_out2';
time=[1501:1500+fn1];
time1=time';
out2 =[ou,time1];
plot(out2(:,2),out2(:,1),'r');
