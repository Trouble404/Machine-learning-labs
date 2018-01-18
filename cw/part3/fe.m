clear all;
data1=xlsread('close3.xlsx');
time = 1:1265;
time = time'
close_data = [data1 , time];


for i=1:1246
  input_matrix(i,:)=close_data(i:i+19);
end
Y=[input_matrix ones(1246,1)];
y=close_data(20:1265,1);


% data settings
N  = 1265; % number of samples
Nu = 100; % number of learning samples
% plot training and validation data
plot(data1,'m-')
grid on, hold on
plot(data1(1:Nu),'b')
plot(data1,'+k','markersize',2)
legend('validation data','training data','sampling markers','location','southwest')
xlabel('time (steps)')
ylabel('y')
%ylim([-.5 1.5])
set(gcf,'position',[1 60 800 400])

% prepare training data
yt = con2seq(data1(1:Nu)');

% prepare validation data
yv = con2seq(data1(Nu+1:end)');

%---------- network parameters -------------
% good parameters (you don't know 'tau' for unknown process)
inputDelays = 1:6:19;  % input delay vector
hiddenSizes = [20 30];   % network structure (number of neurons)

% nonlinear autoregressive neural network
%net = narnet(inputDelays, );
net = narnet(inputDelays, hiddenSizes);
%net = narnet(inputDelays, hiddenSizes);
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
%net = closeloop(net);
% view closeloop version of a net
%view(net);

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
plot(Nu:N-1,Yp,'r')
plot(Nu:N-1,e,'g')
legend('validation data','training data','sampling markers',...
  'prediction','error','location','southwest')
