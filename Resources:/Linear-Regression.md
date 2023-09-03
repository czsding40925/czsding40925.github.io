---
layout: default
title: My Project
---

# Simple Linear Regression Example 1 #

Imagine a scenario where $x$ is the number of hours a student studies and $y$ is the score they get on a test. Given the simple linear model $$
Y=\beta_0+\beta_1x+\varepsilon
$$

With the error term $\varepsilon$ being normally distributed with variance $\sigma^2$. First, consider the following data:
| Hours of Study (x) | Test Score (y) |
|---------------------|----------------|
| 2.0                 | 64.2           |
| 3.0                 | 69.7           |
| 5.0                 | 73.4           |
| 3.0                 | 67.5           |
| 6.0                 | 81.2           |
| 5.0                 | 75.8           |
| 9.0                 | 94.3           |
| 7.0                 | 84.4           |
| 8.0                 | 92.2           |
| 2.0                 | 58.1           |
| 5.0                 | 76.5           |
| 8.0                 | 89.9           |
Let's walk through the following computations: 

Let's setup some preliminary terms first: Here is a list of what we need: 

* $n$: Sample Size 
* $\bar{x}$: Average hours of study $x$
* $\bar{y}$: Average test scores $y$.
* $S_{xx} = \sum_ (x_i-\bar{x})^2 $
* $S_{xy} = \sum_ (y_i-\bar{y})^2 $
* $S_{xy} = \sum (x_i-\bar{x})(y_i-\bar{y})$

Now we can compute them using numpy: 


```python
import numpy as np

# Data
x = np.array([2.0, 3.0, 5.0, 3.0, 6.0, 5.0, 9.0, 7.0, 8.0, 2.0, 5.0, 8.0])
y = np.array([64.2, 69.7, 73.4, 67.5, 81.2, 75.8, 94.3, 84.4, 92.2, 58.1, 76.5, 89.9])

# Sample Size
n = len(x) 
# Calculate the means
x_bar = np.mean(x)
y_bar = np.mean(y)

# Calculate S_xx, S_xy
S_xy = np.sum((x - x_bar) * (y - y_bar))
S_xx = np.sum((x - x_bar)**2) 
S_yy = np.sum((y - y_bar)**2)
print(f"n = {n}")
print(f"x̄ = {x_bar}")
print(f"ȳ = {y_bar:.2f}")
print(f"S_xy = {S_xy:.2f}")
print(f"S_xx = {S_xx}")
print(f"S_yy = {S_yy:.2f}")
```

    n = 12
    x̄ = 5.25
    ȳ = 77.27
    S_xy = 300.40
    S_xx = 64.25
    S_yy = 1447.53


## Computing $\hat{\beta_0}$ and $\hat{\beta_1}$ ##
Recall that the following formula: 
$$
\begin{align*}
    \hat{\beta_1}&=\frac{S_{xy}}{S_{xx}}=\frac{\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^n (x_i-\bar{x})^2}\\
    \hat{\beta_0}&=\bar{y}-\hat{\beta_1}\bar{x}
\end{align*}
$$
Then let's compute: 


```python
# Compute beta_1
hat_beta_1 = S_xy / S_xx

# Compute beta_0
hat_beta_0 = y_bar - hat_beta_1 * x_bar

print(f"β_0 = {hat_beta_0:.2f}")
print(f"β_1 = {hat_beta_1:.2f}")

```

    β_0 = 52.72
    β_1 = 4.68


We get the following: 
* $\hat{\beta_0}=52.72$
* $\hat{\beta_1}=4.68$

And the fitted line is $$\hat{y}=52.72+4.68x$$

Let's visualize this: 


```python
import matplotlib.pyplot as plt

# Calculate the line of best fit
y_pred = hat_beta_0 + hat_beta_1 * x

# Plot the scatter plot of data points
plt.scatter(x, y, color='blue', label='Data Points')

# Plot the line of best fit
plt.plot(x, y_pred, color='red', label='Line of Best Fit')

# Adding labels and title
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Scatter Plot with Line of Best Fit')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

```


    
![png](output_6_0.png)
    


## Sum of Squares Error (SSE) ##
Recall that such a process minimizes **sum of squares error**, or **SSE**. To visiualize, refer to the plot above: the sum of vertical distances between the each blue dots and its corresponding value on the red line is minimized. There are both calculus (using partial derivates) and linear algebra (using inner products and orthogonality) approaches to prove that it is true. But for now, we will just introduce SSE for later computations: $$\text{SSE}=\sum_{i=1}^n (y_i-\hat{y}_i)^2=\sum_{i=1}^n [y_i-(\hat{\beta_0}+\hat{\beta_1}x_i)]^2$$

In addition, we define **residuals** as the vectors of differences between predicted and actual values. We now compute: 


```python
# Residuals
residuals = y - y_pred

# Estimate of error variance
SSE = np.sum(residuals**2)
print(f"SSE = {SSE:.2f}")
```

    SSE = 43.01


## Estimate $\sigma^2$ ## 
Recall that $\sigma^2$ is the variance of the error term $\varepsilon$ where $E(\varepsilon)=0$ by assumption. For an unbiased estimator $S^2$ for $\sigma^2$, we have $$S^2=(\frac{1}{n-2})\sum(Y_i-\hat{Y_i})^2$$The computation is fairly simple. The result we get for $S^2$ is 4.30. 


```python
# Estimate of error variance
sigma_squared = SSE / (len(x) - 2)

print(f"Estimated error variance (σ^2) = {sigma_squared:.2f}")
```

    Estimated error variance (σ^2) = 4.30


## Compute the Variance of $\beta$ ##

Recall that $\hat{\beta_0}$ and $\hat{\beta_1}$ are unbiased estimators. Which means $$E(\hat{\beta_i})=\hat{\beta_i}$$We now want to find the variance for those estimators. We have the following formula: $$V(\hat{\beta_1})=\frac{\sigma^2}{S_{xx}}$$
and  $$V(\hat{\beta_0})=\sigma^2(\frac{1}{n}+\frac{\bar{x}^2}{S_{xx}})=\frac{\sigma^2\sum x_i^2}{nS_{xx}}$$

Now we can compute:


```python
# Variance of beta_1
var_hat_beta_1 = sigma_squared / S_xx

# Variance of beta_0
var_hat_beta_0 = sigma_squared * (1/len(x) + x_bar**2/S_xx)

print(f"Variance of β_1 = {var_hat_beta_1:.2f}")
print(f"Variance of β_0 = {var_hat_beta_0:.2f}")
```

    Variance of β_1 = 0.07
    Variance of β_0 = 2.20


And $V(\hat{\beta_1})=0.07$ and $V(\hat{\beta_0})=2.20$

# Inferences Regarding Parameters $\hat{\beta}$ # 
Let's run some hypothesis tests. We want to see if the data present sufficient evidence to indicate that the slope differs from 0. Test using $\alpha=0.05$ and give bounds using attained significance level. 

We have $H_0:\beta_1=0$ and $H_1:\beta_1\neq 0$ as our null and alternative hypothesis. The formula for the Test statistic is 
$$
T=\frac{\hat{\beta_i}-\beta_{i0}}{S\sqrt{1/S_{xx}}}
$$
We will determine the critical $t$-value with $\alpha=0.05$ and $n-2$ degrees of freedom. 


```python
# t-statistic
t_statistic = hat_beta_1 / (np.sqrt(sigma_squared/S_xx))
print(f"t-statistic = {t_statistic:.2f}")
```

    t-statistic = 18.07


In addition, we can compute the critical $t$-value and $p$-value. 


```python
import scipy.stats as stats

# Critical t-value for two-tailed test with alpha=0.05 and n-2 degrees of freedom
alpha = 0.05
df = len(x) - 2  # degrees of freedom
t_critical = stats.t.ppf(1 - alpha/2, df)

# Compute the p-value
p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df))

print(f"Critical t-value = ±{t_critical:.2f}")
print(f"p-value = {p_value:.4f}")

```

    Critical t-value = ±2.23
    p-value = 0.0000


Alternatively, let's try python's built-in analysis:


```python
import statsmodels.api as sm

# Add a constant to our x values (for the intercept)
X = sm.add_constant(x)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the summary, which will include the t-test for beta_1 and many other statistics
print(model.summary())

```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.970
    Model:                            OLS   Adj. R-squared:                  0.967
    Method:                 Least Squares   F-statistic:                     326.6
    Date:                Fri, 01 Sep 2023   Prob (F-statistic):           5.77e-09
    Time:                        21:16:59   Log-Likelihood:                -24.686
    No. Observations:                  12   AIC:                             53.37
    Df Residuals:                      10   BIC:                             54.34
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         52.7204      1.484     35.516      0.000      49.413      56.028
    x1             4.6755      0.259     18.071      0.000       4.099       5.252
    ==============================================================================
    Omnibus:                        0.986   Durbin-Watson:                   2.589
    Prob(Omnibus):                  0.611   Jarque-Bera (JB):                0.492
    Skew:                          -0.475   Prob(JB):                        0.782
    Kurtosis:                       2.715   Cond. No.                         14.6
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


    /Users/connording40925/.pyenv/versions/3.9.1/lib/python3.9/site-packages/scipy/stats/_stats_py.py:1806: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=12
      warnings.warn("kurtosistest only valid for n>=20 ... continuing "


To construct the confidence interval, we have $$\hat{\beta_i}\pm t_{\alpha/2}S\sqrt{1/S_{xx}}$$
The confidence interval can be found above. We can compute them using the formula as well, and the values indeed match.


```python
# Confidence Interval
CI_lower = hat_beta_1 - t_critical * np.sqrt(sigma_squared/S_xx)
CI_upper = hat_beta_1 + t_critical * np.sqrt(sigma_squared/S_xx)
print(f"95% Confidence Interval for β1: ({CI_lower:.2f}, {CI_upper:.2f})")

```

    95% Confidence Interval for β1: (4.10, 5.25)


## Inferences Regarding Linear Functions of the Model Parameters: Simple Linear Regression ##
We want to find the confidence interval for the average score of students who studied 6 hours. 

Recall that the confidence interval for $E(Y)$ for $x=x^*$, where $x^*$ is a particular value of $x$, can be written as $$\hat{\beta_0}+\hat{\beta_1}x^*\pm t_{\alpha/2}S\sqrt{\frac{1}{n}+\frac{(x^*-\bar{x})^2}{S_{xx}}}$$where $t_{\alpha/2}$ is based on $n-2$ degrees of freedom. Here comes the computation: 


```python
# Predicted value for x_0 = 6
x_0 = 6
y_hat = hat_beta_0 + hat_beta_1 * x_0

# Critical t-value for two-tailed test with alpha=0.10 and n-2 degrees of freedom
alpha_90 = 0.10
t_critical_90 = stats.t.ppf(1 - alpha_90/2, df)

# Calculate SE_y_hat
SE_y_hat = np.sqrt(sigma_squared * (1/len(x) + (x_0 - x_bar)**2 / np.sum((x - x_bar)**2)))

# 90% Confidence Interval for E(Y) when x=6
CI_lower_90 = y_hat - t_critical_90 * SE_y_hat
CI_upper_90 = y_hat + t_critical_90 * SE_y_hat

print(f"90% Confidence Interval for E(Y) when x=6: ({CI_lower_90:.2f}, {CI_upper_90:.2f})")

```

    90% Confidence Interval for E(Y) when x=6: (79.63, 81.91)


### Let's run it using statsmodels as well ###


```python
import numpy as np
import statsmodels.api as sm

# Add constant to predictor (for intercept term in the regression equation)
X = sm.add_constant(x)

# Fit model
model = sm.OLS(y, X).fit()

# Compute 90% confidence interval for E(Y) when x=6
x_new = np.array([[1, 6]])  # 1 for the constant term (intercept) and 6 for the value of x
prediction = model.get_prediction(x_new)
ci = prediction.conf_int(alpha=0.10)

print(f"90% Confidence Interval for E(Y) when x=6: ({ci[0][0]:.2f}, {ci[0][1]:.2f})")

```

    90% Confidence Interval for E(Y) when x=6: (79.63, 81.91)


## Predicting a Particular Value of $Y$ ## 

Suppose a student from the same sampling pool studied for 8.5 hours. Predict the student's score $90\%$ prediction interval. 

Recall that The $100(1-\alpha)\%$ prediction for $Y$ when $x=x^*$ is $$\hat{\beta_0}+\hat{\beta_1}x^*\pm t_{\alpha/2}S\sqrt{1+\frac{1}{n}+\frac{(x^*-\bar{x})^2}{S_{xx}}}$$

Now we compute: 


```python
from scipy.stats import t
# Prediction
x_0 = 8.5
y_0_hat = hat_beta_0 + hat_beta_1 * x_0

# Prediction interval
alpha = 0.10
t_value = t.ppf(1 - alpha/2, df=n-2)
PI_low = y_0_hat - t_value * np.sqrt(sigma_squared * (1 + 1/n + (x_0 - x_bar)**2 / np.sum((x - x_bar)**2)))
PI_high = y_0_hat + t_value * np.sqrt(sigma_squared * (1 + 1/n + (x_0 - x_bar)**2 / np.sum((x - x_bar)**2)))

print(f"Computational 90% Prediction Interval: ({PI_low:.2f}, {PI_high:.2f})")

```

    Computational 90% Prediction Interval: (88.26, 96.66)


### Using Statsmodel ### 


```python
# Using statsmodels
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
x_new = np.array([[1, 8.5]])  # 1 for the constant term and 8.5 for the value of x
prediction = model.get_prediction(x_new)
pi = prediction.conf_int(alpha=0.10)
print(f"Statsmodels 90% Prediction Interval: ({pi[0][0]:.2f}, {pi[0][1]:.2f})")
```

    Statsmodels 90% Prediction Interval: (90.59, 94.33)


## Correlation ## 
Naturally, we wonder whether there is a correlation between the hours a student has studied and his/her test score. To do so, we make an inference on $\rho$, which is the **correlation coefficient**. So we make the null hypothesis $H_0: \rho=0$ and alternative $H_\alpha: \rho\neq 0$. Recall that the test statistic for $\rho$ is $$t=\frac{r\sqrt{n-2}}{\sqrt{1-r^2}}$$where $$r=\frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}}=\hat{\beta_1}\sqrt{\frac{S_{xx}}{S_{yy}}}$$In addition, we can compute the **coefficient of determination** $r^2$: 


```python
# Compute r
r = S_xy/np.sqrt(S_xx * S_yy) 

# Compute test statistic
t_stat = r * np.sqrt((n-2)/(1-r**2))

# Compute p-value (two-tailed)
p_value = 2 * (1 - t.cdf(abs(t_stat), df=n-2))

# Compute the coefficient of determination
r_squared = r ** 2 

# Results
print("Sample Correlation Coefficient (r):", round(r, 2))
print(f"R^2 (Coefficient of Determination): {round(r_squared, 2)}")
print("T-test Statistic:", round(t_stat, 2))
print("P-value:", round(p_value, 2))
```

    Sample Correlation Coefficient (r): 0.99
    R^2 (Coefficient of Determination): 0.97
    T-test Statistic: 18.07
    P-value: 0.0


Then we can run it using statsmodel:


```python
# Add a constant term for intercept
X = sm.add_constant(x)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display summary statistics
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.970
    Model:                            OLS   Adj. R-squared:                  0.967
    Method:                 Least Squares   F-statistic:                     326.6
    Date:                Fri, 01 Sep 2023   Prob (F-statistic):           5.77e-09
    Time:                        21:16:59   Log-Likelihood:                -24.686
    No. Observations:                  12   AIC:                             53.37
    Df Residuals:                      10   BIC:                             54.34
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         52.7204      1.484     35.516      0.000      49.413      56.028
    x1             4.6755      0.259     18.071      0.000       4.099       5.252
    ==============================================================================
    Omnibus:                        0.986   Durbin-Watson:                   2.589
    Prob(Omnibus):                  0.611   Jarque-Bera (JB):                0.492
    Skew:                          -0.475   Prob(JB):                        0.782
    Kurtosis:                       2.715   Cond. No.                         14.6
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


    /Users/connording40925/.pyenv/versions/3.9.1/lib/python3.9/site-packages/scipy/stats/_stats_py.py:1806: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=12
      warnings.warn("kurtosistest only valid for n>=20 ... continuing "


## Linear Regression Example 2 ##
Let's consider a scenario where you're trying to predict the price of houses based on various factors:

* $x_1$: Square footage of the house
* $x_2$: Number of bedrooms
* $x_3$: Age of the House
* $x_4$: Distance from the city center (in miles)
Let's simulate some data for this scenario:

1. Square Footage ($x_1$):Varies between 800 and 4000 square feet.
2. Number of bedrooms ($x_2$): Ranges between 1 and 5.
3. Age of the house ($x_3$): Ranges between 0 and 50 years.
4. Distance from the city center ($x_4$): Ranges between 0.5 and 20 miles.

We'll simulate the data for 1000 houses. We'll create the data such that the house prices have a linear relationship with the predictors, but we'll also add some noise to make it more realistic:


```python
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Generate the data
n = 1000
x0 = np.ones(n)
x1 = np.random.uniform(800, 4000, n)
x2 = np.random.randint(1, 6, n)
x3 = np.random.randint(0, 51, n)
x4 = np.random.uniform(0.5, 20, n)

# True coefficients
b0 = 50000  # base price
b1 = 50    # price per square foot
b2 = 15000 # price per bedroom
b3 = -500  # decrease in price per year due to age
b4 = -1000 # decrease in price per mile from city center

# Simulate the house prices
noise = np.random.normal(0, 10000, n) # Some noise
y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + noise

# Remember to tranpose to column vector. 
Y = y.transpose()
```

## Basics in Fitting Linear Model in Matrices ## 
Remember a general linear model is in the form of 
$$
Y=\beta_0+\beta_1x_1+\dots+\beta_kx_k+\varepsilon
$$
Suppose we make $n$ independent observations. Then each of $y_1,y_2,\dots, y_n$ can be written as 
$$
y_i=\beta_0+\beta_1x_{i1}+\dots+\beta_kx_{ik}+\varepsilon_i
$$
Remember that we can represent them in the matrix form. Here are the matrices we are interested in: 
$$
\begin{align*}
    \mathbf{Y}&=
    \begin{bmatrix}
        y_1\\
        y_2\\
        \vdots\\
        y_n
    \end{bmatrix}
    \mathbf{X}=
    \begin{bmatrix}
        x_0&x_{11}&x_{12}&\dots&x_{1k}\\
        x_0&x_{21}&x_{22}&\dots&x_{2k}\\
        \vdots&\vdots&\vdots&\vdots&\vdots\\
        x_0&x_{n1}&x_{n2}&\dots&x_{nk}
    \end{bmatrix}\\
    \mathbf{\beta}&=
    \begin{bmatrix}
        \beta_0\\
        \beta_1\\
        \vdots\\
        \beta_k
    \end{bmatrix}
    \mathbf{\varepsilon}=
    \begin{bmatrix}
        \varepsilon_1\\
        \varepsilon_2\\
        \vdots\\
        \varepsilon_n
    \end{bmatrix}
\end{align*}
$$
Thus, we can write the linear model of $n$ linear equations conviniently using matrix form: $$\mathbf{Y}=\mathbf{X}\mathbf{\beta}+\mathbf{\varepsilon}$$Now, let's recall that the estimator $\hat{\beta}$ can be written as 
$$
\hat{\beta}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}
$$This can be shown using techniques in linear algebra. 

## Compute $\hat{\beta}$ ## 


```python
X_t = np.array([x0, x1, x2, x3, x4])  
X = X_t.transpose()

hat_B = (np.linalg.inv(X_t.dot(X))).dot(X_t.dot(Y))
print(hat_B)
```

    [ 5.15566137e+04  4.97946904e+01  1.48777135e+04 -5.14082061e+02
     -1.01711829e+03]


## Running the Model Using statsmodels ##


```python
import pandas as pd
import statsmodels.api as sm

# Create a DataFrame
df = pd.DataFrame({'SquareFootage': x1, 'Bedrooms': x2, 'Age': x3, 'DistanceFromCity': x4, 'Price': y})

# Fit a linear model
X = df[['SquareFootage', 'Bedrooms', 'Age', 'DistanceFromCity']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.961
    Model:                            OLS   Adj. R-squared:                  0.960
    Method:                 Least Squares   F-statistic:                     6074.
    Date:                Fri, 01 Sep 2023   Prob (F-statistic):               0.00
    Time:                        21:16:59   Log-Likelihood:                -10655.
    No. Observations:                1000   AIC:                         2.132e+04
    Df Residuals:                     995   BIC:                         2.134e+04
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    const             5.156e+04   1364.833     37.775      0.000    4.89e+04    5.42e+04
    SquareFootage       49.7947      0.350    142.356      0.000      49.108      50.481
    Bedrooms          1.488e+04    227.620     65.362      0.000    1.44e+04    1.53e+04
    Age               -514.0821     22.348    -23.004      0.000    -557.936    -470.228
    DistanceFromCity -1017.1183     57.652    -17.642      0.000   -1130.251    -903.985
    ==============================================================================
    Omnibus:                        0.227   Durbin-Watson:                   2.079
    Prob(Omnibus):                  0.893   Jarque-Bera (JB):                0.311
    Skew:                          -0.013   Prob(JB):                        0.856
    Kurtosis:                       2.918   Cond. No.                     1.07e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.07e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.


## Finding SSE and Estimating the $\sigma^2$ ##

The SSE is given by 
$$
\mathbf{Y}'\mathbf{Y}-\hat{\beta}'\mathbf{X}'\mathbf{Y}
$$
In addition, an unbiased estimator $S^2$ for $\sigma^2$, the variance of the error term, is given by 
$$
\text{SSE}/[n-(k+1)]
$$


```python
#Finding SSE
SSE = y.dot(Y)-hat_B.transpose().dot(X_t.dot(Y)) 
S_squared = SSE / (n - 5 - 1)
SSE, S_squared
```




    (105202106180.23438, 105837128.95395812)



### Find using statsmodel ###


```python
import statsmodels.api as sm

# Design matrix X
X_star = np.column_stack([x0, x1, x2, x3, x4])

# Fit the OLS model
model = sm.OLS(Y, X_star)
results = model.fit()

# Extract SSE
sse_statsmodels = results.ssr

# Extract variance estimate for error term
var_estimate_statsmodels = sse_statsmodels / (n - 5 - 1)

sse_statsmodels, var_estimate_statsmodels
```




    (105202106180.35088, 105837128.95407532)



## Inferences Concerning the Parameters ## 
Find a $90\%$ interval for the average $(E(Y))$ price of a 10-year-old house that is 2000 square feet with 2 bedrooms and 3 miles away from the city center.

Recall that the confidence interval is given by 
$$\mathbf{a}'\hat{\beta}\pm t_{\alpha/2}S\sqrt{\mathbf{a}'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{a}}
$$
Here $\mathbf{a}$ is a specific vector of $x$. In this case, 
$$
\mathbf{a}=
\begin{bmatrix}
1 \\
2000 \\
2 \\
10 \\
3
\end{bmatrix}
$$


```python
# Finding Confidence Interval
import scipy.stats as stats
alpha = 0.05
a = np.array([1,2000,2,10,3]) 
t_value = stats.t.ppf(1 - alpha/2, df=n-5)
X_inv = np.linalg.inv(np.dot(X.T, X))
CI_upper = a.transpose().dot(hat_B) + t_value * np.sqrt(S_squared * a.transpose().dot(X_inv.dot(a)))
CI_lower = a.transpose().dot(hat_B) - t_value * np.sqrt(S_squared * a.transpose().dot(X_inv.dot(a)))
CI_lower, CI_upper
```




    (171366.5398358419, 174051.9521071141)




```python
# Using statsmodels
new_a = a.reshape(1,-1)
prediction = results.get_prediction(new_a)
ci_statsmodels = prediction.conf_int(alpha=0.05)
ci_statsmodels
```




    array([[171367.21473216, 174051.2772108 ]])



## Predicting a Particular Value of $Y$ ##
Find a $90\%$ prediction interval for a particular price of a 10-year-old house that is 2000 square feet with 2 bedrooms and 3 miles away from the city center. Note the subtle difference from the previous example (average price vs. particular price). However, the formula is mostly the same: 
$$
\mathbf{a}'\hat{\beta}\pm t_{\alpha/2}S\sqrt{1+\mathbf{a}'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{a}}
$$


```python
# Finding Prediction Interval 
CI_upper = a.transpose().dot(hat_B) + t_value * np.sqrt(S_squared * (1+a.transpose().dot(X_inv.dot(a))))
CI_lower = a.transpose().dot(hat_B) - t_value * np.sqrt(S_squared * (1+a.transpose().dot(X_inv.dot(a))))
CI_lower, CI_upper
```




    (152476.53091828062, 192941.96102467537)




```python
# Using Statsmodels 
pi_statsmodels = prediction.conf_int(obs=True, alpha=0.05)
pi_statsmodels
```




    array([[152486.70066762, 192931.79127534]])



## Multiple Coefficient of Determination ##
We would like to know how much information of the variation of housing prices $Y$ can be explained by the factors listed above. Just like in the simple linear model case, we have a multiple coefficient of determination $R^2$, given by 
$$
R^2=\frac{S_{yy}-\text{SSE}}{S_{yy}}
$$



```python
# multiple coefficient of determination 
X = np.column_stack((x0, x1, x2, x3, x4))

# Fit the regression model
model = sm.OLS(y, X).fit()
y_bar = np.mean(y) 
y_hat = model.predict(X) # this should have been done a while ago lol but I'm too lazy so I just use model.predict
# In practice, just perform a matrix multiplication/dot product
SSE = np.sum((y - y_hat)**2)
SST = np.sum((y - np.mean(y))**2)
R_squared =  1 - (SSE/SST) # Equivaluent to the formula above 

# Using statsmodels to compute R^2
R2_statsmodels = model.rsquared

R_squared, R2_statsmodels
```




    (0.9606557702089958, 0.9606557702089958)



## Model Reduction ##
We wonder whether some independent variables actually played a role in housing factors: for example, whether the number of rooms and how far away it is from the city center actually have an effect on the housing prices. Which means we are trying to test the null hypothesis 
$$
\beta_2=\beta_4=0
$$
If true, then the linear model can be reduced to 
$$
y=\beta_0+\beta_1x_1+\beta_3x_3
$$
In addition, we introduce some terminologies: 
* $\text{SSE}_C$ is the SSE for the complete model.
* $\text{SSE}_R$ is the SSE for the reduced model.
Note that large values of $$\text{SSE}_R-\text{SSE}_C$$ would lead us to reject the null hypothesis (since this is the variance explained by the variables that were left out). In fact,$(\text{SSE}_R-\text{SSE}_C)/\sigma^2$, $\text{SSE}_R/\sigma^2$, $\text{SSE}_C/\sigma^2$ all possess $\chi^2$ distribution with $(\text{SSE}_R-\text{SSE}_C)/\sigma^2$, $\text{SSE}_C/\sigma^2$ being independent. Therefore, the test statistic

$$
F=\frac{(\text{SSE}_R-\text{SSE}_c)/(k-g)}{\text{SSE}_C/(n-[k+1])}
$$
follows an $F$-distribution with $v_1=k-g$ numerator degrees of freedom and $v_2=n-(k+1)$ denominator degrees of freedom, where $k$ is the total number of independent variables and $g$ is the number of independent variables in the reduced model. Let's try to compute $F$:



```python
# Too lazy again, code from ChatGPT 
import numpy as np
import statsmodels.api as sm

# Sample data
np.random.seed(42)
n = 1000
x0 = np.ones(n)
x1 = np.random.uniform(800, 4000, n)
x2 = np.random.randint(1, 6, n)
x3 = np.random.randint(0, 51, n)
x4 = np.random.uniform(0.5, 20, n)
b0 = 50000
b1 = 50
b2 = 15000
b3 = -500
b4 = -1000
noise = np.random.normal(0, 10000, n)
y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + noise
X = np.column_stack((x0, x1, x2, x3, x4))

# Fit the full model
model_full = sm.OLS(y, X).fit()

# Fit the reduced model (without x2 and x4)
X_reduced = np.column_stack((x0, x1, x3))
model_reduced = sm.OLS(y, X_reduced).fit()

# Compute the F-statistic manually
q = 2
SSE_R = np.sum(model_reduced.resid**2)
SSE_F = np.sum(model_full.resid**2)
F_manual = ((SSE_R - SSE_F)/q) / (SSE_F/(n - X.shape[1]))

# Use statsmodels for the F-test
hypothesis = '(x2 = 0), (x4 = 0)'
f_test = model_full.f_test(hypothesis)
F_statsmodels = f_test.fvalue

F_manual, F_statsmodels

```




    (2333.7176129609174, 2333.7176129609174)



## Let's say $\alpha=0.1$ ##
See comparison below. Null hypothesis is rejected 


```python
from scipy.stats import f

alpha = 0.1
df1 = 2
df2 = n - 5 - 1  # 5 predictors (including intercept) and minus 1
F_critical = f.ppf(1 - alpha, df1, df2)
if F_statsmodels > F_critical:
    print("Reject the null hypothesis")
else:
    print("Do not reject the null hypothesis")

```

    Reject the null hypothesis

