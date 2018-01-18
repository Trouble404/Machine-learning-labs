% Set up your own data in the following form
% X: N by 2 matrix of data
% y: class labels -1 or +1
% include column of ones for bias
N=1000;
C=[2 1;1 2];
X=randn(N,2);
m1 = [0 2];
X1 = X + kron(ones(N,1),m1);
A=chol(C);
Y1=X1*A;

plot(Y1(:,1),Y1(:,2),'mx');
hold on;

X=randn(N,2);
m2 = [1.5 0];
X1 = X + kron(ones(N,1),m2);
A=chol(C);
Y2=X1*A;
plot(Y2(:,1),Y2(:,2),'bx');
hold on;

X1 = [Y1 ones(N,1)];% Gaussian with mean = [0 2] is class 1
X2 = [Y2 ones(N,1)];
X2(1:N,3)=-1;% Gaussian with mean = [1.5 0] is class -1
X=[X1;X2];% combine two matrixs into one
%set the class
Y=ones(2*N,1);
Y(N+1:2*N,1)=-1;
% Separate into training and test sets (check: >> doc randperm)
ii = randperm(2*N);%disorganize the sort
Xtr = X(ii(1:N),:);%traning class 1
ytr = Y(ii(1:N),:);%test class 1
Xts = X(ii(N+1:2*N),:);%traning class -1
yts = Y(ii(N+1:2*N),:);%test class -1

% initialize weights
w = randn(3,1);
% Error correcting learning
eta = 0.002;
for iter=1:N
j = ceil(rand*N);% random choose
if ( (ytr(j)*Xtr(j,:))*w < 0 )%learning judgement
w = w + eta*ytr(j)*Xtr(j,:)';%weight adjustment
end
end
% plotting
n1=linspace(-6,6,50);
n2=-(w(1)/w(2))*n1-w(3)/w(2);
plot(n1,n2)
yhts = Xts*w;
PercentageError = (size(find(yts.*yhts < 0),1))/(N);%number of uncorrect weight

disp(PercentageError);
grid on;