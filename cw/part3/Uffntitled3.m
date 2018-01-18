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
%��������
hiddenLayerSize = 20; %���ز���Ԫ����
net = fitnet(hiddenLayerSize);

% �������ϣ�����ѵ�������Ժ���֤���ݵı���
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%ѵ������
[net,tr] = train(net,input2',targets');

%% ����ͼ���ж���Ϻû�
yn=net(input2');
errors=targets-yn;
figure, ploterrcorr(errors)                      %������������������20lags��
%figure, parcorr(errors)                          %����ƫ������
%[h,pValue,stat,cValue]= lbqtest(errors)         %Ljung��Box Q���飨20lags��
%figure,plotresponse(con2seq(targets),con2seq(yn))   %��Ԥ���������ԭ����
%figure, ploterrhist(errors)                      %���ֱ��ͼ
figure, plotperform(tr)                          %����½���

%% ����Ԥ������Ԥ�⼸��ʱ���
fn=500;  %Ԥ�ⲽ��Ϊfn
n=length(data1);
lag=20;
f_in=data1(n-fn-lag: n-fn-1)';
kk=f_in;
f_out=zeros(1,fn);  %Ԥ�����
% �ಽԤ��ʱ���������ѭ�������������������
for i=1:fn
    f_out(i)=net(f_in');
    term = f_out(i);
    f_in=[f_in(2:end),f_out(i)];
end
% ����Ԥ��ͼ
ou = f_out';
time=[759:758+fn];
time1=time';
out =[ou,time1];
plot(out(:,2),out(:,1),'r');
hold on;
plot(out(:,2),data1(759:758+fn),'b');
%figure,plot(1:1501,iinput,'b',1501:2001,[iinput(end),f_out],'r')