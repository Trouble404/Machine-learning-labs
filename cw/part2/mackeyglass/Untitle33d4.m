clear all
run('mackeyglass.m')
x=X;
% �ýű�������NAR������Ԥ��
lag=20;    % �Իع����
iinput=x;    % xΪԭʼ���У���������
n=length(iinput);

%׼��������������
inputs=zeros(lag,n-lag);
for i=1:n-lag
    inputs(:,i)=iinput(i:i+lag-1)';
end
input1=inputs';
input2=input1(1:1481,:);
input3=input2;
targets=x(1:1481);

%���Իع�
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
f_out1=f_out1';%Ԥ�����
% �ಽԤ��ʱ���������ѭ�������������������
for i=1:481
    f_in1=input3(1481-1+i: 1500-1+i);
    term1=[f_in1 ones(1,1)];
    f_out1(i)=term1*w;
    f_in1=[f_in1(2:end),f_out1(i)];
end
% ����Ԥ��ͼ
ou1 = f_out1;
time4=[1501:1981];
time5=time4';
out1 =[ou1,time5];
plot(out1(:,2),out1(:,1),'r');
hold on;
plot(out1(:,2),X(1501:1981),'b');

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

f_in=X(n-fn-lag: n-fn-1)';
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
time=[1501:1500+fn];
time1=time';
out =[ou,time1];
plot(out(:,2),out(:,1),'r');
hold on;
plot(out(:,2),X(1501:1500+fn),'b');
%figure,plot(1:1501,iinput,'b',1501:2001,[iinput(end),f_out],'r')

%% ����Ԥ������Ԥ�⼸��ʱ���
fn1=1000;  %Ԥ�ⲽ��Ϊfn

f_in2=X(n-fn1-lag: n-fn1-1)';
f_out2=zeros(1,fn1);  %Ԥ�����
% �ಽԤ��ʱ���������ѭ�������������������
for i=1:fn1
    f_out2(i)=net(f_in2');
    term = f_out2(i);
    f_in2=[f_in2(2:end),f_out2(i)];
end
% ����Ԥ��ͼ
ou = f_out2';
time=[1501:1500+fn1];
time1=time';
out2 =[ou,time1];
plot(out2(:,2),out2(:,1),'r');
