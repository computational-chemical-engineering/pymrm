#!/bin/bash

# The commands provided here will be executed for the unit test step in the ci
# but may also be used locally
# execute from root folder!

python -m pip install -e ".[test]"
pytest --junitxml=test/pytest-results.xml --cov=. --import-mode=importlib --cov-report=html --cov-report=xml:test/coverage.xml --cov-config=.coveragerc
