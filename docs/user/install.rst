Installation
===================

.. contents:: Table of Contents

Install Python
--------------
recsyslearn is officially tested on Python from version 3.10. If you do not have at least this python
version installed, go to the official `Python download page <https://www.python.org/downloads/>`_.

Install recsyslearn
-------------------
There are 2 different ways in which you could install recsyslearn, depending on the branch release
you would like to use.

Stable release
--------------

To install recsyslearn, run this command in your terminal:

.. code-block:: console

    $ pip install recsyslearn

This is the preferred method to install recsyslearn, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

From sources
------------

The sources for recsyslearn can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/giuliowaitforitdavide/recsyslearn

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/giuliowaitforitdavide/recsyslearn/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/giuliowaitforitdavide/recsyslearn
.. _tarball: https://github.com/giuliowaitforitdavide/recsyslearn/tarball/master


Verifying Installation
----------------------
After installing recsyslearn, you can verify that it has been successfully installed
by running the following command on your favourite terminal/command prompt:

.. code-block:: bash

    python3 -c "import recsyslearn; print(recsyslearn.__version__)"

You should see the following output:

.. parsed-literal::
    |version|

Congratulations! Your machine has recsyslearn and you're now ready to
create your first experiment!



What's next?
------------
Start creating your first experiment by following the :doc:`quickstart` guide.