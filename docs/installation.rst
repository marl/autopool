.. _installation:

Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

pypi
~~~~
The simplest way to install *autopool* is through the Python Package Index
(PyPI). This will ensure that all required dependencies are fulfilled.
This can be achieved by executing the following command::

    pip install autopool

or::

    sudo pip install autopool

to install system-wide, or::

    pip install -u autopool

to install just for your own user.

Source
~~~~~~

If you've downloaded the archive manually from the `releases
<https://github.com/marl/autopool/releases/>`_ page, you can install using the
`setuptools` script::

    tar xzf autopool-VERSION.tar.gz
    cd autopool-VERSION/
    python setup.py install

If you intend to develop autopool or make changes to the source code, you can
install with `pip install -e` to link to your actively developed source tree::

    tar xzf autopool-VERSION.tar.gz
    cd autopool-VERSION/
    pip install -e .

Alternately, the latest development version can be installed via pip::

    pip install git+https://github.com/marl/autopool
