 x = rand(1000,1); % generate 1000 random numbers within 0 and 1
 hist(x,40); % generate the frequency histogram for 40 ranges.
% the frequency histogram has equally 10 ranges and returns
% value of eachbin to nn and the center value of each bin to xx
 [nn, xx] = hist(x);
 bar(nn); % separate each bin.