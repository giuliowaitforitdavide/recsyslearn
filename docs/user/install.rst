Installation
============

Stable release
--------------

recsyslearn is published on PyPI as pre-built wheels for Linux (glibc and musl, x86-64
and ARM64), macOS (Intel and Apple Silicon) and Windows (x86-64). It supports Python
|min_python| and later:

.. code-block:: console

    $ pip install recsyslearn

or, with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: console

    $ uv add recsyslearn

No Rust toolchain is needed: the wheels contain the compiled core.

.. |min_python| replace:: 3.11

From source
-----------

On other platforms pip builds the package from the source distribution, which requires
a `Rust toolchain <https://rustup.rs>`_ (1.85 or later). The same applies to installing
the development version:

.. code-block:: console

    $ pip install git+https://github.com/giuliowaitforitdavide/recsyslearn

Verifying the installation
--------------------------

.. code-block:: console

    $ python -c "import recsyslearn; print(recsyslearn.__version__)"

should print |version|.

Next, read about :doc:`the input tables <data>` or jump to the :doc:`quickstart`.
