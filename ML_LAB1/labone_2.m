 x = randn(1000,1);% generate 1000 random number drawn from a Gaussian 
                   % distribution of mean=0 and variance=1.
 
  hist(x,40);
 help hist
 [nn, xx] = hist(x);
 bar(nn);