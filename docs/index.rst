.. autopool documentation master file, created by
   sphinx-quickstart on Thu Apr 26 11:17:42 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. |br| raw:: html

   <br />

AutoPool
========

AutoPool is an adaptive pooling function with a learnable parameter that allows
it to smoothly interpolate between min-, mean-, softmax- and max-pooling.
It is implemented as `Keras <https://keras.io/>`_ or `Tensorflow-Keras <https://www.tensorflow.org/api_docs/python/tf/keras>`_ layers for easy integration in deep learning architectures.

AutoPool was originally developed for multiple instance learning from weakly
labeled time-series data as detailed in the
`AutoPool paper <https://arxiv.org/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Getting Started
===============
.. toctree::
    :maxdepth: 2

    installation
    definitions


API Documentation
=================
.. toctree::
   :maxdepth: 3

   api


Examples
========
.. toctree::
    :maxdepth: 3

    examples


Contribute
==========
- `Issue tracker <http://github.com/marl/autopool/issues>`_
- `Source code <http://github.com/marl/autopool>`_


Changes
=======
.. toctree::
   :maxdepth: 1

   changes



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
