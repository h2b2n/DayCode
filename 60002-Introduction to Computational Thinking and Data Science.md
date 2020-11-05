# LECTURE 6 : Monte Carlos Simulation
#### 1. Key Concepts
> popularity  
> sample  
> variance  
> confidence level & interval  
> law of large numbers <-> gambler's fallacy  
> regession to the mean  
> PDF(Probability Density Function)  


#### 2. Additional Reading
> :https://machinelearningmastery.com/monte-carlo-sampling-for-probability/  
> :https://www.investopedia.com/terms/m/montecarlosimulation.asp  

#### 3. Useful Functions 
```python
  import numpy as np
  from scipy import stats
  
  cnt = int(input())
  ar = list(map(int, input().rstrip().split()))
  print(format(np.mean(ar),".1f"))
  print(format(np.median(ar),".1f"))
  mode =stats.mode(ar)
  print(mode[0])
  ##print(format(np.var(ar), ".1f"))
  print(format(np.std(ar), ".1f"))
  print(format(np.mean(ar)-1.96*(np.std(ar)/np.sqrt(cnt)), ".1f")+" "+format(np.mean(ar)+1.96*(np.std(ar)/np.sqrt(cnt)), ".1f"))
``` 

# Lecture 7 : Confidence Intervals
#### 1. Key Concepts
> Assumptions underlying Empirical Rule, the mean estimation error is zero.
```python
  random.hauss(0,100)
  # or PDF -  derivative of cumulative distribution function
  def gaussian(x, mu, sigma) :
  fac1 = (1.0/sigma*((2*pylab.pi)**05)))
  fac2 =  pylab.e**-((x-mu)**2)/(2*sigma**2))
  return fac1*fac2
  # x-axis -> y-axis
```
> Scipy library 
```python
  for numStd in (1, 1.96, 3) :
    area = Scipy.integrate.quad(gaussian, \
                                mu-numStd*sigma, mu+numStd*sigma,\
                                (mu, sigma))[0] ## [0] estimate of the integral  
```
> The Central Limit Theoren(CLT)
>> Given a sufficiently large sample :
>> 1. The means of the samples in a set of samples will be approximately normally distributed
>> 2. It will have a mean close to the mean of the population 
>> 3. The variance of the sample means will be close to the variance of the pop. divided by the sample size.

