dvhastats
=========

|logo|


|build| |pypi| |Docs| |lgtm| |lgtm-cq| |Codecov|

A library of prediction and statistical process control tools. Although based
on work in `DVH Analytics <http://www.dvhanalytics.com>`__, all tools in this
library are generic and not specific to radiation oncology. See
our `documentation <http://dvha-stats.readthedocs.io>`__ for advanced uses.

What does it do?
----------------

* Read data from CSV, supply as numpy array or dict
* Basic plotting
   * Simple one-variable plots from data
   * Control Charts (Univariate, Multivariate, & Risk-Adjusted)
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
-  `PTable <https://github.com/kxxoling/PTable>`__


Basic Usage
------------

.. code-block:: python

    from dvhastats.ui import DVHAStats
    s = DVHAStats("your_data.csv")  # use s = DVHAStats() for test data

    >>> s.var_names
    ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    >>> s.show('V1')  # or s.show(0), can provide index or var_name

|plot|


Multivariate Control Chart (w/ non-normal data)
###############################################
.. code-block:: python

    ht2_bc = s.hotelling_t2(box_cox=True)
    >>> ht2_bc.show()

|hotelling-t2-bc|


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
.. |lgtm-cq| image:: https://img.shields.io/lgtm/grade/python/g/cutright/DVHA-Stats.svg?logo=lgtm&label=code%20quality
   :target: https://lgtm.com/projects/g/cutright/DVHA-Stats/context:python
   :alt: lgtm code quality
.. |lgtm| image:: https://img.shields.io/lgtm/alerts/g/cutright/DVHA-Stats.svg?logo=lgtm
   :target: https://lgtm.com/projects/g/cutright/DVHA-Stats/alerts
   :alt: lgtm
.. |Codecov| image:: https://codecov.io/gh/cutright/DVHA-Stats/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/cutright/DVHA-Stats
   :alt: Codecov
.. |Docs| image:: https://readthedocs.org/projects/dvha-stats/badge/?version=latest
   :target: https://dvha-stats.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |plot| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/91908372-0c4c2d80-ec71-11ea-9dfc-7c4f6c209542.png" width="350 alt="Basic Plot"/></a>

.. |hotelling-t2-bc| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/91908394-179f5900-ec71-11ea-88a0-9c95d714fb4c.png" width="350 alt="Multivariate Control Chart w/ Box Cox Transformation"/></a>

.. |pca| raw:: html

    <a><img src="https://user-images.githubusercontent.com/4778878/92050205-16922880-ed52-11ea-9967-d390577380b6.png" width="350 alt="Principal Component Analysis"/></a>


.. |logo| raw:: html

    <a>
      <img src="https://user-images.githubusercontent.com/4778878/92505112-351c7780-f1c9-11ea-9b5c-0de1ad2d131d.png" width='400' alt="DVHA logo"/>
    </a>
