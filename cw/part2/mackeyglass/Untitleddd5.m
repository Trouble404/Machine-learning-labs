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
% data settings
N  = 2000; % number of samples
Nu = 1500; % number of learning samples
plot(X,'m-')
grid on, hold on
plot(X(1:Nu),'b')
plot(X,'+k','markersize',2)
legend('validation data','training data','sampling markers','location','southwest')
xlabel('time (steps)')
ylabel('y')
ylim([-.5 1.5])
set(gcf,'position',[1 60 800 400])

% prepare training data
yt = con2seq(X(1:Nu)');

% prepare validation data
yv = con2seq(X(Nu+1:end)');

%---------- network parameters -------------
% good parameters (you don't know 'tau' for unknown process)
inputDelays = 1:6:19;  % input delay vector
%hiddenSizes = [6 3];   % network structure (number of neurons)
hiddenSizes = 20;
%-------------------------------------

% nonlinear autoregressive neural network
%net = narnet(inputDelays, );
net = narnet(inputDelays, hiddenSizes);

% [Xs,Xi,Ai,Ts,EWs,shift] = preparets(net,Xnf,Tnf,Tf,EW)
%
% This function simplifies the normally complex and error prone task of
% reformatting input and target timeseries. It automatically shifts input
% and target time series as many steps as are needed to fill the initial
% input and layer delay states. If the network has open loop feedback,
% then it copies feedback targets into the inputs as needed to define the
% open loop inputs.
%
%  net : Neural network
%  Xnf : Non-feedback inputs
%  Tnf : Non-feedback targets
%   Tf : Feedback targets
%   EW : Error weights (default = {1})
%
%   Xs : Shifted inputs
%   Xi : Initial input delay states
%   Ai : Initial layer delay states
%   Ts : Shifted targets
[Xs,Xi,Ai,Ts] = preparets(net,{},{},yt);

% train net with prepared training data
net = train(net,Xs,Ts,Xi,Ai);
% view trained net
view(net)

% close feedback for recursive prediction
net = closeloop(net);
% view closeloop version of a net
view(net);

% prepare validation data for network simulation
yini = yt(end-max(inputDelays)+1:end); % initial values from training data
% combine initial values and validation data 'yv'
[Xs,Xi,Ai] = preparets(net,{},{},[yini yv]);

% predict on validation data
predict = net(Xs,Xi,Ai);

% validation data
Yv = cell2mat(yv);
% prediction
Yp = cell2mat(predict);
% error
e = Yv - Yp;

% plot results of recursive simulation
figure(1)
plot(Nu:N,Yp,'r')
plot(Nu:N,e,'g')
legend('validation data','training data','sampling markers',...
  'prediction','error','location','southwest')