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



# Lecture 9 : Understanding Experimental Data
#### 1. Key Concepts
> Given gathered data, develop theory and design a computation. 
>> Firstly, plot the data by using pylab
>> Secondly, fit curves to Data 
>>        A. Objective Function 
>>        B. Find the curve that minimizes the sum of the vertical distances.
>>        That's because we want to predict dependent Y value for given independent X value. The vertical distance indicates error in the prediction. Thus, minimize the least squares of objective function, equal to minimize the variance. 
>>        Then how to ? 
>>        Linear Regression to find a polynomial representation with pylab.ployfit .
```python 
  def fitData(fileName) :
    xVals, yVals = getData(fileName)
    xVals = pylab.array(xVals)
    yVals = pylab.array(yVals)
    pylab.plot(xVals, yVals, 'bo')
    a, b = pylab.polyfit(xVals, yVals,1) # 1 - best line if 2, best parabola
    estYVals = a*pylab.array(xVals) + b 
    pylab.plot(xVals, estYVals,'r--')
    model = pylab.polyfit(xVals, yVals,1)
    pylab.plot(xVals, pylab.polyval(model, xVals), 'g', label = 'Linear fit, k= '+ str(round(1/model[0],5)))    
```
