# mvShapiroTest

## Shapiro-Wilk Test for Multivariate Normality and Skew-Normality

**Blanca Monroy-Castillo and Elizabeth Gonzalez-Estrada** 

blancamonroy.96@gmail.com, egonzalez@colpos.mx

Colegio de Postgraduados, México


Parametric statistical methods assume a specific model for the probability distribution of the observations. When non-robust methods are used, statistical inferences may be invalid if the assumed distribution is not a plausible model for the  observations.   Goodness of fit tests are the statistical methods used for testing distributional assumptions on a data set.

This package contains tests specially developed for testing goodness of fit of the multivariate normal and multivariate skew-normal distributions when parameters are unknown.

The following tests are implemented.

An extension of the Shapiro-Wilk test for testing goodness of fit of the multivariate skew normal distribution  and a test based  on both a closure property of the skew normal distributions and the canonical transformation of the multivariate skew normal distribution (González-Estrada et al. 2022). 

A generalization of the Shapiro-Wilk test for testing multivariate normality  (Villasenor and Gonzalez-Estrada, 2009).

The package also provides graphical tools for assessing the fit of the pdf's studied in the above papers.


## Initialization

 ```
 pip install mvShapiroTest
 ```
 
 ### Modules
 
 The package provides two modules, *test* and *graphs*. There are three ways to call the modules.
 
 First, import all modules with the code 
 
 ```
 import mvShapiroTest
 ```
 
 Also, it is possible import a unique module with the code
 
 ```
 from mvShapiroTest import test
 from mvShapiroTest import graphs
 ```
 ### Functions
 
 #### test Module
 
 The functions in the **test module** are
 
  - mvshapiro(X) [A generalization of Shapiro-Wilk test for multivariate normality (Villasenor and Gonzalez-Estrada, 2009).]
  
      Where **X** is a numeric data matrix with d columns and n rows. This could be an array or data frame.
      Sample size (n) must be larger than vector dimension (d).
      When d = 1, `mvshapiro(X)` produces the same results as `shapiro.test(X)`. 
      
      The **values** returned by `mvshapiro(X)` are:
      
      - *statistic*  the value of the generalized Shapiro-Wilk statistic for testing multivariate normality. 
      - *p_value*     an approximated p-value of the test.
      - *Method*      the character string "Generalized Shapiro-Wilk test for multivariate normality".

  - canonical(y, xi, Omega, alpha) [Canonical transformation of a random variable with multivariate skew-normal distribution (Capitanio, 2012).]
  
    Where the arguments are:
      - **y** is a numeric data matrix with d columns and n rows. This could be an array or data frame.
              Sample size (n) must be larger than vector dimension (d).
      - **xi**, **Omega** and **alpha** are location, scale matrix and slant parameters.
      
      Azzalini and Dalla Valle (1996)'s multivariate skew-normal distribution is considered.
         
      The value returned by this function is a numeric data matrix with d columns and n rows. One column has a standard skew-normal distribution and the remaining d-1 columns have standard normal distribution. The coordinates of the canonical form are independent (Capitanio, 2012).
      
   -  mvsn_shapiro(y, method = "EM", R_HOME = None) [Shapiro-Wilk test for the multivariate skew-normal distribution (González-Estrada et al. 2022).]   
 
       Where the arguments are:
      
      - **y** is a numeric data matrix with d columns and n rows. This could be an array or data frame. Sample size (n) must be larger than vector dimension (d).
      -  **method** is the parameter estimation method used, there are two available: expectation maximization "EM" (by default) and maximum likelihood "MLE" implemented in the **sn** package of R software (R Core Team, 2022). If method = "MLE" then it is necessary to have R and the **sn** library (Azzalini, 2022) installed. 
      -  **R_HOME** when "MLE" method is used, the R_HOME directory must be provided. This directory is got with *R.home(component = "home")* command from R console.  
      
      Note: It is must be installed rpy2 when method = "MLE" (`pip install rpy2`).   

      The **values** returned by `mvsn_shapiro(y, method = "EM", R_HOME = None)` are:
      
      - *statistic*  the value of the Shapiro-Wilk statistic for testing multivariate skew normality. 
      - *p_value*     an approximated p-value of the test.
      - *Method*      the character string "Shapiro-Wilk test for multivariate skew normal distributions".

   - mvsn_test(y, method = "EM", R_HOME = None)[Test for the multivariate skew-normal distribution based on the closure property of sums of normal and skew normal distributions (González-Estrada et al. 2022). Simulation results indicate that this test has a good control of the type I error probability.]
    
     Where the arguments are like in `mvsn_shapiro` function.
     
      The **values** returned by `mvsn_test(y, method = "EM", R_HOME = None)` are:
      
      - *statistic*  the value of the test statistic for  multivariate skew normality. 
      - *p_value*     an approximated p-value of the test.
      - *Method*      the character string "Test for multivariate skew normal distributions based on a closure property".

  
 #### graphs module
   - plot(y, dist = "MVSN", pdf = True, ecdf = True, bins = 20) [Diagnostic plots for checking the fit of probability distributions to data.]
    
      Where the arguments are:
    
      - **y** is a numeric data matrix with d columns and n rows. 
      - dist  is the theoretical distribution of the data. Available options are MVSN (default) for the multivariate skew-normal distribution,  and MVN for the multivariate normal distribution. 
      - pdf = TRUE, returns the histogram of relative frequencies and the fitted theoretical  distribution.
      - ecdf = TRUE, returns the empirical distribution function and the fitted theoretical cumulative distribution function.
      - bins is the number of bins of the histogram when pdf = TRUE.


## Examples

Testing  multivariate normality on the famous iris virginica data set.

```
import mvShapiroTest
import numpy as np
from mvShapiroTest import test, graphs
from sklearn import datasets
iris = datasets.load_iris()

data = np.array(iris.data[100:150, :])  # iris virginica
test = test.mvshapiro(data)  # Generalized Shapiro Wilk test for multivariate normality

pvalue = test['p_value'] 

graphs.plot(data)
```

Computing the canonical transformation of the iris virginica data set.
```
from mvem.stats import multivariate_skewnorm as mvsn
from numpy.linalg import eig

xi_fitted, Omega_fitted, lmbda_fitted = mvsn.fit(data, return_loglike = False, ftol = 1e-10) # parameter estimation by EM method
eigenv_Omega, e_vec_Omega = eig(Omega_fitted) 
e_vec_Omega = np.matrix(e_vec_Omega)
mat = np.diag(1/np.sqrt(eigenv_Omega))
sqrt_inv_Omega = np.dot(e_vec_Omega, mat)
sqrt_inv_Omega = np.dot(sqrt_inv_Omega, np.transpose(e_vec_Omega))  
alpha_fitted = np.dot(sqrt_inv_Omega, lmbda_fitted)  # estimation of the slant parameter
test.canonical(data, xi = xi_fitted, Omega = Omega_fitted, alpha = alpha_fitted)   # canonical transformation
``` 
 
## References

Azzalini, A. and Dalla Valle, A. (1996). The multivariate skew-normal distribution. Biometrika 83: 715-726.

Azzalini, A. (2022). The R package 'sn': The Skew-Normal and Related Distributions such as the Skew-t and the SUN (version 2.0.2). URL http://azzalini.stat.unipd.it/SN/,https://cran.r-project.org/package=sn

Capitanio, A. (2012). On the canonical form of scale mixtures of skew-normal distributions. arXiv:1207.0797. URL
https://doi.org/10.48550/arXiv.1207.0797

González-Estrada, E. et al. (2022). Shapiro-Wilk test for multivariate skew-normality. Computational Statistics. https://doi.org/10.1007/s00180-021-01188-y

Villasenor, J.A. and Gonzalez-Estrada, E. (2009). A generalization of Shapiro-Wilk's test for multivariate normality. Communications in Statistics: Theory and Methods, 38 11, 1870-1883. http://dx.doi.org/10.1080/03610920802474465

Villasenor, J.A. and Gonzalez-Estrada, E. (2015). A variance ratio test of fit for Gamma distributions. Statistics and Probability Letters, 96 1, 281-286. http://dx.doi.org/10.1016/j.spl.2014.10.001
