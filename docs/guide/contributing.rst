Contributing
============

We welcome contributions! Here's how to get started.

Development Setup
----------------

.. code-block:: bash

   git clone https://github.com/PlumBlossomMaid/PaddleOcean.git
   cd PaddleOcean
   pip install -e .

Running Tests
------------

.. code-block:: bash

   pip install pytest pytest-timeout
   pytest tests/ -v --timeout=120

Code Style
---------

This project uses Ruff for linting and formatting.

.. code-block:: bash

   pip install ruff
   ruff check .
   ruff format .

Pull Request Process
-------------------

1. Fork the repo and create your branch from ``main``.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code lints.
5. Issue the pull request!

Building Documentation
---------------------

.. code-block:: bash

   cd docs
   pip install -r requirements-docs.txt
   make html
   # open _build/html/index.html
