<a>
  <img src="https://user-images.githubusercontent.com/4778878/30754005-b7a7e808-9f86-11e7-8b0f-79d1006babdf.jpg" width='480' alt="DVHA logo"/>
</a>

---------  
# DVHA Stats
A library of prediction and statistical process control tools. Although based 
on work in [DVH Analytics](http://dvhanalytics.com), all statistical tools in 
this library are generic and not radiation oncology.

[![build](https://github.com/cutright/DVHA-Stats/workflows/build/badge.svg)](https://github.com/cutright/DVHA-Stats/actions)
<a href="https://pypi.org/project/dvha-stats/">
  <img src="https://img.shields.io/pypi/v/dvha-stats.svg" alt="PyPi Version" /></a>
<a href="https://lgtm.com/projects/g/cutright/DVHA-Stats/context:python">
  <img src="https://img.shields.io/lgtm/grade/python/g/cutright/DVHA-Stats.svg?logo=lgtm&label=code%20quality" alt="LGTM Code Quality" /></a>
<a href="https://codecov.io/gh/cutright/DVHA-Stats">
  <img src="https://codecov.io/gh/cutright/DVHA-Stats/branch/master/graph/badge.svg" />
</a>

### What does it do?
* Read data from CSV or supply as numpy array 
* Plotting
    * Simple one-variable plots from data
    * Control Charts (Univariate and Multivariate)
    * Heat Maps (correlations, PCA, etc.)
* Perform Box-Cox transformations
* Calculate Correlation matrices
* Perform Multi-Variable Linear Regressions
* Perform Principal Component Analysis (PCA)

### Coming Soon:
- [ ] Multi-Variable Regression residual and quantile plots
- [ ] Backward-elimination for Multi-Variable Linear Regressions
- [ ] Risk-Adjusted Control Charts using Multi-Variable Linear Regressions
- [ ] Machine learning regressions based on scikit-learn


**NOTE**: This project is brand new and very much under construction.

Source-Code Installation
---------
~~~
pip install dvha-stats
~~~
or
~~~
pip install git+https://github.com/cutright/DVHA-Stats.git
~~~
Or clone the project and run:
~~~
python setup.py install
~~~

Dependencies
---------
* [Python](https://www.python.org) >3.5
* [SciPy](https://scipy.org)
* [NumPy](http://numpy.org)
* [Scikit-learn](http://scikit-learn.org)
* [regressors](https://pypi.org/project/regressors/)
* [matplotlib](http://matplotlib.org/)

### Initialize and Plot Data
~~~
>>> from dvhastats.stats import DVHAStats
>>> s = DVHAStats("tests/testdata/multivariate_data.csv")
>>> s.var_names
['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
>>> s.show('V1')  # or s.show(0), can provide index or var_name
~~~
<img src='https://user-images.githubusercontent.com/4778878/91908372-0c4c2d80-ec71-11ea-9dfc-7c4f6c209542.png' align='center' width='350' alt="Data Plot">

### Correlation Matrix
~~~
>>> pearson_mat = s.correlation_matrix()
>>> pearson_mat.show()
~~~
<img src='https://user-images.githubusercontent.com/4778878/92064453-1ea69400-ed63-11ea-8f72-5034c577c1e3.png' align='center' width='350' alt="Pearson-R Correlation Matrix">

Like-wise, a Spearman correlation matrix:
~~~
>>> spearman_mat = s.correlation_matrix("Spearman")
>>> spearman_mat.show()
~~~
<img src='https://user-images.githubusercontent.com/4778878/92177010-4a7a5600-ee05-11ea-91b9-2a0128eafe5b.png' align='center' width='350' alt="Spearman Correlation Matrix">


### Univariate Control Chart
~~~
>>> ucc = s.univariate_control_charts()
>>> ucc["V1"].show()  # or ucc[0].show(), can provide index or var_name
~~~
<img src='https://user-images.githubusercontent.com/4778878/91908380-0fdfb480-ec71-11ea-9394-d029a8a6727e.png' align='center' width='350' alt="Univariate Control Chart">

### Hotelling T^2
Example to calculate a Multivariate Control Chart with Hotelling T^2 values
~~~
>>> ht2 = s.hotelling_t2()
>>> ht2.show()
~~~

<img src='https://user-images.githubusercontent.com/4778878/91908391-166e2c00-ec71-11ea-941b-321e01f56542.png' align='center' width='350' alt="Multivariate Control Chart">

### Hotelling T^2 with Box-Cox Transformation
Example to calculate the Hotelling T^2 values and apply a Box-Cox transformation
~~~
>>> ht2_bc = s.hotelling_t2(box_cox=True)
>>> ht2_bc.show()
~~~

<img src='https://user-images.githubusercontent.com/4778878/91908394-179f5900-ec71-11ea-88a0-9c95d714fb4c.png' align='center' width='350' alt="Multivariate Control Chart with Box-Cox Transformation">

### Principal Component Analysis (PCA)
~~~
>>> pca = s.pca()
>>> pca.show()
~~~
<img src='https://user-images.githubusercontent.com/4778878/92050205-16922880-ed52-11ea-9967-d390577380b6.png' align='center' width='350' alt="PCA Feature Heat Map">