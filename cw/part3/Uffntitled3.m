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
input2=input_matrix;
targets=data1(20:1259);
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
n=length(data1);
lag=20;
f_in=data1(n-fn-lag: n-fn-1)';
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
time=[759:758+fn];
time1=time';
out =[ou,time1];
plot(out(:,2),out(:,1),'r');
hold on;
plot(out(:,2),data1(759:758+fn),'b');
%figure,plot(1:1501,iinput,'b',1501:2001,[iinput(end),f_out],'r')