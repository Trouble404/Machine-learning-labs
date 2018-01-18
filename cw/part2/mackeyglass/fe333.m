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
Y=[input_matrix,ones(1481,1)];
%%Best linear predictor
N  = 2000; % number of samples
Nu = 1501; % number of learning samples
y=X;
plot(y,'m-')
grid on, hold on
plot(y(1:Nu),'b')
plot(y,'+k','markersize',2)
legend('validation data','training data','sampling markers','location','southwest')
xlabel('time (steps)')
ylabel('y')
ylim([-.5 1.5])
set(gcf,'position',[1 60 800 400])

% prepare training data
yt = con2seq(y(1:Nu)');

% prepare validation data
yv = con2seq(y(Nu+1:end)');

%---------- network parameters -------------
% good parameters (you don't know 'tau' for unknown process)
inputDelays = 1:6:19;  % input delay vector
hiddenSizes = [6 3];   % network structure (number of neurons)
%-------------------------------------

% nonlinear autoregressive neural network
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
plot(Nu+1:N,Yp,'r')
plot(Nu+1:N,e,'g')
legend('validation data','training data','sampling markers',...
  'prediction','error','location','southwest')