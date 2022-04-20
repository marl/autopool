#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import numpy as np
import autopool.keras


@pytest.mark.parametrize('init', ['zeros', 'ones'])
def test_autopool_keras(init):

    # Create a random array
    x = np.random.randn(1, 3, 10)**2

    # pool over the trailing dimension
    ap_layer = autopool.keras.AutoPool1D(kernel_initializer=init, axis=-1)

    x_ap = ap_layer(x)

    assert x_ap.shape == (1, 3)

    # Compute by force with numpy
    if init == 'zeros':
        # For zero-init, the default state will be a normal average
        scaled = np.zeros_like(x)
    elif init == 'ones':
        # For ones-init, we should have the same behavior as softmax pooling
        scaled = x

    max_val = np.max(scaled, axis=-1, keepdims=True)
    softmax = np.exp(scaled - max_val)
    weights = softmax / np.sum(softmax, axis=-1, keepdims=True)

    x_np_ap = np.sum(x * weights, axis=-1, keepdims=False)

    assert np.allclose(x_np_ap, x_ap), (x, x_ap, x_np_ap)


def test_softmaxpool_keras():
    # Create a random array
    x = np.random.randn(1, 3, 10)**2

    # pool over the trailing dimension
    ap_layer = autopool.keras.SoftMaxPool1D(axis=-1)

    x_ap = ap_layer(x)

    assert x_ap.shape == (1, 3)

    # Compute by force with numpy
    max_val = np.max(x, axis=-1, keepdims=True)
    softmax = np.exp(x - max_val)
    weights = softmax / np.sum(softmax, axis=-1, keepdims=True)

    x_np_ap = np.sum(x * weights, axis=-1, keepdims=False)

    assert np.allclose(x_np_ap, x_ap), (x, x_ap, x_np_ap)

