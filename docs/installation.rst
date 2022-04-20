.. _installation:

Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^

pypi
~~~~
The simplest way to install *autopool* is through the Python Package Index
(PyPI). This will ensure that all required dependencies are fulfilled.
This can be achieved by executing the following command::

    python -m pip install autopool[keras]

which installs autopool for the Keras backend.
If you would rather use tensorflow, you can do the following::

    python -m pip install autopool[tf]

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
    python -m pip install -e .

Alternately, the latest development version can be installed via pip::

    python -m pip install git+https://github.com/marl/autopool
