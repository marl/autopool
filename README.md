# autopool
Adaptive pooling operators for Multiple Instance Learning ([documentation](http://autopool.readthedocs.io/)).

[![PyPI](https://img.shields.io/pypi/v/autopool.svg)](https://pypi.python.org/pypi/autopool)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/marl/autopool.svg?branch=master)](https://travis-ci.org/marl/autopool)
[![Coverage Status](https://coveralls.io/repos/github/marl/autopool/badge.svg?branch=master)](https://coveralls.io/github/marl/autopool?branch=master)
[![Documentation Status](https://readthedocs.org/projects/autopool/badge/?version=latest)](http://scaper.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/autopool.svg)]()


AutoPool is an adaptive (trainable) pooling operator which smoothly interpolates between common pooling operators, such
as min-, max-, or average-pooling, automatically adapting to the characteristics of the data.

AutoPool can be readily applied to any differentiable model for time-series label prediction. AutoPool is presented in the following paper, where it is evaluated in conjunction with convolutional neural networks for Sound Event Detection:

[Adaptive pooling operators for weakly labeled sound event detection](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/mcfee_autopool_taslp_2018.pdf)<br/>
Brian Mcfee, Justin Salamon, and Juan Pablo Bello<br/>
IEEE Transactions on Audio, Speech and Language Processing, In press, 2018.



