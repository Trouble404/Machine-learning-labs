N = 1000;
x1 = zeros(N,1);% create a 1000*1 matrix with 0
 for n=1:N
     % fill the matrix with two uniform random distribute arrays
 x1(n,1) = sum(rand(12,1))-sum(rand(12,1));
end
 hist(x1,100); % generate the frequency histogram for 100 ranges.