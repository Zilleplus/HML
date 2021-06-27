# If it takes one hour to train a Decision Tree on a training set containing 1 million instances, roughtly how much time will it take to train another Decision Tree on a training set containing 10 million instances.

if: 
    m = number of samples
    n = tree depth ~= log(m)

If on average every split, divides the samples 50/50. And in order to split we have about m evaluations. Then we can say that the computational complexity is about m*log(m).

[m*10*log(10*m)]/[m*log(m)] 
= [m*10*(log(10)+ log(m))]/[m*log(m)] 
= [10*(log(10) + log(m))]/log(m)
= (10*log(10))/log(m) + 10

if m = 1e6

log(10)~=3
log(1e6)~= 20

then -> (10*3)/20 = 30/20 + 10 = 11.5


 Increasing the number of samples by a factor or 10, then n increase by log(10)~=3 and m by a factor of 10.

Total: About 30 times increase in calculation time.

Appendix: The appendix has about the same thing, they come up with 11.7
