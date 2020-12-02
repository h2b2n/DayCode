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
###### Assumptions underlying Empirical Rule, the mean estimation error is zero.
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
###### Given gathered data, develop theory and design a computation.  
> Firstly, plot the data by using pylab  
> Secondly, fit curves to Data  
>>        A. Objective Function  
>>        B. Find the curve that minimizes the sum of the vertical distances.  
>>        That's because we want to predict dependent Y value for given independent X value. The vertical distance indicates error in the prediction. Thus, minimize the least squares of objective function, equal to minimize the variance.  
>        Then how to ?  
>        Linear Regression to find a polynomial representation with pylab.ployfit .  
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
>> Extra Reading on Function 'polyval' :  
>> https://dellwindowsreinstallationguide.com/python-numpy-and-matplotlib-curve-fitting-using-polyfit-and-polyval/  
###### Criteria to measure how good a model is.  
> 1. Comparing to each other by using function 'aveMeanSquareError(yVals, estYVals)'  
> 2. R-squared, coefficient of determination, to get a sense of absolute goodness of fit.  
```python 
  def RSquared(observed, predicted) :
    error = ((predicted - observed)**2).sum()
    meanError = error/len(observed)
    return 1-(meanError/numpy.var(observed))
    # considering R-square = SSE/variance 
    # Errors in estimates / variability in measured data 
```
>> 0 < R-Square < 1 relating to the realtionship between the values predicted by the model and actual data.  

# Lecture 10. (cont.) 
#### 1. Key Concepts  
###### Training and Test error 
> Training Error: The classification error of a model *on the same data, called trainging data.*  
> Test Error: The classification error calculated by *another data, test dataset,* which have the y-values as well.      
> https://rapidminer.com/blog/validate-models-training-test-error/  
###### Cross Validate : generate models using one dataset and use it on another dataset  
> To get how the model was validated
> When fitting the one training data with noise and non-perfction, we can face over-fitting.  
> https://scikit-learn.org/stable/modules/cross_validation.html  
###### Leave-one-out Cross Validation - K-fold
> repeating random sampling
```python 
  def splitData(xVals, yVals) :
    toTrain = random.sample(range(len(xVals)), len(xVals)//2)
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(xVals)):
      if i in toTrain :
        trainX.append(xVals[i])
        trainY.append(yVals[i])
      else :
        testX.append(xVals[i])
        testY.append(yVlas[i])
    return trainX, trainY, testX, testY
  
  for f in range(numSubsets) :
    trainX, trainY, testX, testY = splitData(xVals, yVals)
    for d in dimensions :
      model = pylab.polyfit(trainX, trainY, d)
      train_estYvals = pylab.polyval(model, trainX)
      test_estYvals = pylab.polyval(model, testX)
      train_rSquared[d].append(rSquared(trainY, train_estYvals))
      test_rSquared[d].append(rSquared(testY, test_estYvals))
  
  for d in dimensions :
    mean = round(sum(test_rSquareds[d])/len(test_rSquared[d]),4)
    sd = round(numpy.std(rSquared[d], 4)    
    
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import KFold
  X_train, X_test, y_train, y_test = train_test_split(xVals, yVals)
  kf = KFold(n_splits=2)
  kf.get_n_splits(X_train)
  KFold(n_splits=2, random_state=None, shuffle=False)
```
> which line explains better than the other?  (Given that multiple trys) 
> Highest average r-squared  
> Smallest deviation across trials  
> Simplest model  
# Lecture 13. Classification
#### 1. Key Concepts
###### a. Performance Measure
> Accuracy  
> Sensitivity(called Recall) = TP/(TP+FN)  
> Specificity(called Precision) = TN/(TN+FP)  
> Positive predictive value = TP/(TP+FP)  
> Negative predictive value = TN/(TN+FN)  
###### ? Data Split ?  
###### b. Logistic Regression
> predicting probability of an event  
```python
  import sklearn.linear_model import LogisticRegression
  #Data Type : Dictionary, 
  #data of training or test DataSet by using d.getFeatures() and d.getLabel()
  LogitsticRegression = sklearn.linear_nodel.LogisticRegression
  model = LogisticRegression().fit(featureVecs, labels)
  
  testFeatureVecs = [d.getFeatures() for d in testDataSet
  probs = model.predict_proba(testFeatureVecs)
  
  #coef?
```  
