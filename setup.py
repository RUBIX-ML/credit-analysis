# MIT License

# Copyright (c) 2019 RUBIX-ML

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import setuptools
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

setup(
    name = 'credit-analysis',
    version = '1.0',
    author = 'Eric Sun',
    author_email = 'eric.sun@wearerubix.com',
    description = 'Credit card payment prediction project',
    long_description = read('README.md'),
    license = 'MIT',
    url = 'https://github.com/RUBIX-ML/credit-analysis',

    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = [
        'graphviz==0.12'
        'matplotlib==3.1.1'
        'numpy==1.16.3'
        'pandas==0.23.4'
        'plotly==4.1.0'
        'scikit-learn==0.19.2'
        'scipy==1.1.0'
        'sklearn==0.0'
        'sklearn-pandas==1.8.0'
        'xgboost==1.0.0'
        'lightgbm==2.2.3' 
    ]
)