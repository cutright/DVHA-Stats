#!/usr/bin/env python
# -*- coding: utf-8 -*-

# setup.py

from setuptools import setup
from dvhastats._version import __version__, __author__, __email__


with open('requirements.txt', 'r') as doc:
    requires = [line.strip() for line in doc]

with open('README.md', 'r') as doc:
    long_description = doc.read()

CLASSIFIERS = [
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Healthcare Industry",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "Development Status :: 2 - Pre-Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Scientific/Engineering :: Physics"]


setup(
    name='dvha-stats',
    version=__version__,
    include_package_data=True,
    python_requires='>3.5',
    packages=['dvhastats'],
    package_dir={'dvhastats': 'dvhastats'},
    description='Simple DICOM tag editor built with wxPython and pydicom',
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    url='https://github.com/cutright/DVHA-Stats',
    download_url='https://github.com/cutright/DVHA-Stats/archive/master.zip',
    license="MIT License",
    keywords=['stats', 'statistical process control', 'control charts'],
    classifiers=CLASSIFIERS,
    install_requires=requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    test_suite='tests',
    tests_require=[]
)
