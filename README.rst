=========
dvhastats
=========

|logo|


|build| |pypi| |Docs| |lgtm| |Codecov|

A library of prediction and statistical process control tools. Although based
on work in `DVH Analytics <http://www.dvhanalytics.com>`__, all statistical
tools in this library are generic and not specific to radiation oncology. See
our `documentation <http://dvha-stats.readthedocs.io>`__ for advanced uses.

What does it do?
----------------
* Read data from CSV or supply as numpy array
* Basic plotting
    * Simple one-variable plots from data
    * Control Charts (Univariate and Multivariate)
    * Heat Maps (correlations, PCA, etc.)
* Perform Box-Cox transformations
* Calculate Correlation matrices
* Perform Multi-Variable Linear Regressions
* Perform Principal Component Analysis (PCA)

Other information
-----------------

-  Free software: `MIT license <https://github.com/cutright/DVHA-Stats/blob/master/LICENSE>`__
-  Documentation: `Read the docs <https://dvha-stats.readthedocs.io>`__
-  Tested on Python 3.6, 3.7, 3.8

Dependencies
------------

-  `scipy <https://scipy.org>`__
-  `numpy <http://www.numpy.org>`__
-  `scikit-learn <http://scikit-learn.org>`__
-  `regressors <https://pypi.org/project/regressors/>`__
-  `matplotlib <http://matplotlib.org>`__


Basic Usage
------------

.. code-block:: python

    from dvhastats.ui import DVHAStats
    s = DVHAStats("tests/testdata/multivariate_data.csv")

    >>> s.var_names
    ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    >>> s.show('V1')  # or s.show(0), can provide index or var_name

|plot|

.. code-block:: python

    >>> s.show(0, plot_type="hist")

|hist|

Pearson-R Correlation Matrix
############################
.. code-block:: python

    pearson_mat = s.correlation_matrix()
    >>> pearson_mat.show()

|pearson|

Spearman Correlation Matrix
###########################
.. code-block:: python

    spearman_mat = s.correlation_matrix("Spearman")
    >>> spearman_mat.show()

|spearman|

Univariate Control Chart
########################
.. code-block:: python

    ucc = s.univariate_control_charts()
    >>> ucc["V1"].show()  # or ucc[0].show(), can provide index or var_name

|control-chart|

Multivariate Control Chart
##########################
.. code-block:: python

    ht2 = s.hotelling_t2()
    >>> ht2.show()

|hotelling-t2|

Multivariate Control Chart (w/ non-normal data)
###############################################
.. code-block:: python

    ht2_bc = s.hotelling_t2(box_cox=True)
    >>> ht2_bc.show()

|hotelling-t2-bc|

Multi-Variable Linear Regression
###############################################
.. code-block:: python

    import numpy as np
    y = np.random.rand(s.observations)
    mvr = s.linear_reg(y)
    >>> mvr.show()

|mvr|

.. code-block:: python

    >>> mvr.show("prob")

|mvr-prob|

Principal Component Analysis (PCA)
##################################
.. code-block:: python

    pca = s.pca()
    >>> pca.show()

|pca|

.. |build| image:: https://github.com/cutright/DVHA-Stats/workflows/build/badge.svg
   :target: https://github.com/cutright/DVHA-Stats/actions
   :alt: build
.. |pypi| image:: https://img.shields.io/pypi/v/dvha-stats.svg
   :target: https://pypi.org/project/dvha-stats
   :alt: PyPI
.. |lgtm| image:: https://img.shields.io/lgtm/grade/python/g/cutright/DVHA-Stats.svg?logo=lgtm&label=code%20quality
   :target: https://lgtm.com/projects/g/cutright/DVHA-Stats/context:python
   :alt: lgtm
.. |Codecov| image:: https://codecov.io/gh/cutright/DVHA-Stats/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/cutright/DVHA-Stats
   :alt: Codecov
.. |Docs| image:: https://readthedocs.org/projects/dvha-stats/badge/?version=latest
   :target: https://dvha-stats.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |plot| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/91908372-0c4c2d80-ec71-11ea-9dfc-7c4f6c209542.png" width="350 alt="Basic Plot"/></a>

.. |hist| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/92502706-e4efe600-f1c5-11ea-9f63-4218899e885b.png" width="350 alt="Basic Histogram"/></a>

.. |pearson| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/92064453-1ea69400-ed63-11ea-8f72-5034c577c1e3.png" width="350 alt="Pearson-R Matrix"/></a>

.. |spearman| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/92177010-4a7a5600-ee05-11ea-91b9-2a0128eafe5b.png" width="310 alt="Spearman Matrix"/></a>

.. |control-chart| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/91908380-0fdfb480-ec71-11ea-9394-d029a8a6727e.png" width="350 alt="Control Chart"/></a>

.. |hotelling-t2| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/91908391-166e2c00-ec71-11ea-941b-321e01f56542.png" width="350 alt="Multivariate Control Chart"/></a>

.. |hotelling-t2-bc| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/91908394-179f5900-ec71-11ea-88a0-9c95d714fb4c.png" width="350 alt="Multivariate Control Chart w/ Box Cox Transformation"/></a>

.. |pca| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/92050205-16922880-ed52-11ea-9967-d390577380b6.png" width="350 alt="Principal Component Analysis"/></a>


.. |logo| raw:: html

    <a>
      <img src="https://user-images.githubusercontent.com/4778878/92505112-351c7780-f1c9-11ea-9b5c-0de1ad2d131d.png" width='400' alt="DVHA logo"/>
    </a>

.. |mvr| raw:: html

    <a>
      <img src="https://user-images.githubusercontent.com/4778878/92635692-dd027580-f29b-11ea-834e-3a05e36498fe.png" width='400' alt="DVHA logo"/>
    </a>

.. |mvr-prob| raw:: html

    <a>
      <img src="https://user-images.githubusercontent.com/4778878/92635704-e390ed00-f29b-11ea-9425-d758e21dcebc.png" width='400' alt="DVHA logo"/>
    </a>
