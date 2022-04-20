# autopool
Adaptive pooling operators for Multiple Instance Learning ([documentation](http://autopool.readthedocs.io/)).

[![PyPI](https://img.shields.io/pypi/v/autopool.svg)](https://pypi.python.org/pypi/autopool)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/autopool/badge/?version=latest)](http://autopool.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/autopool.svg)]()


AutoPool is an adaptive (trainable) pooling operator which smoothly interpolates between common pooling operators, such
as min-, max-, or average-pooling, automatically adapting to the characteristics of the data.

AutoPool can be readily applied to any differentiable model for time-series label prediction. AutoPool is presented in the following paper, where it is evaluated in conjunction with convolutional neural networks for Sound Event Detection:

[Adaptive pooling operators for weakly labeled sound event detection](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/mcfee_autopool_taslp_2018.pdf)<br/>
Brian Mcfee, Justin Salamon, and Juan Pablo Bello<br/>
IEEE Transactions on Audio, Speech and Language Processing, In press, 2018.

Installation
------------

To install the keras-based implementation:
```
python -m pip install autopool[keras]
```
For the tensorflow implementation:
```
python -m pip install autopool[tf]
```

Definition
----------
AutoPool extends softmax-weighted pooling by adding a trainable parameter α to be learned jointly with all other trainable  model parameters:

<img src="https://user-images.githubusercontent.com/3009670/43347985-d3bcc072-91c5-11e8-8074-f9b064d7f5a3.png" width="500px">

Here, `p(Y|x)` denotes the prediction for an *instance* `x`, and `X` denotes a set (bag) of instances.  The aggregated prediction `P(Y|X)` is a weighted average of the instance-level predictions.
Note that when α = 0 this reduces to an unweighted mean; when α = 1 this simplifies to soft-max pooling; and when α → ∞ this approaches the max operator. Hence the name: AutoPool.

Usage
-----
AutoPool is implemented as a [keras](https://keras.io/) layer, so using it is as straightforward as using any standard Keras pooling layer, for example:

```
from autpool.keras import AutoPool1D
bag_pred = AutoPool1D(axis=1)(instance_pred)
```

Further details and examples are provided in the [documentation](http://autopool.readthedocs.io/).


Constrained and Regularized AutoPool
------------------------------------
In the [paper](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/mcfee_autopool_taslp_2018.pdf) we show there can be benefits to either constraining the range α can take, or, alternatively, applying l2 regularization on α; this results in constrained AutoPool (CAP) and regularized AutoPool (RAP) respectively. Since AutoPool is implemented as a [keras](https://keras.io/) layer, CAP and RAP can be can be achieved through the layer's optional arugments:

CAP with non-negative α:
```
bag_pred = AutoPool1D(axis=1, kernel_constraint=keras.constraints.non_neg())(instance_pred)
```

CAP with α norm-constrained to some value `alpha_max`:
```
bag_pred = AutoPool1D(axis=1, kernel_constraint=keras.constraints.max_norm(alpha_max, axis=0))(instance_pred)
```
Heuristics for determining sensible values of `alpha_max` are given in the paper (section III.E).

RAP with l2 regularized α:
```
bag_pred = AutoPool1D(axis=1, kernel_regularizer=keras.regularizers.l2(l=1e-4))(instance_pred)
```

CAP and RAP can be combined, of course, by applying both a kernel constraint and a kernel regularizer.

If using the tensorflow-based implementation, all of the above will also work, except that `keras` should be replaced by
`tensorflow.keras`.

Multi-label
-----------
AutoPool directly generalizes to multi-label settings, in which multiple class labels may be applied to each instance x (for example "car" and "siren" may both be present in an instance). In this setting, a separate auto-pooling operator is applied to each class. Rather than a single parameter α, there is a vector of parameters α_c where c indexes the output vocabulary. This allows a jointly trained model to adapt the pooling strategies independently for each category. Please see the [paper](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/mcfee_autopool_taslp_2018.pdf) for further details.
