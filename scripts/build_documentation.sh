#!/bin/bash

# The commands provided here will be executed for building the documentation in the ci
# but may also be used locally

pip install myst-parser # workaround. Should be included in Docker image?
pip install -e .
cd docs/sphinx
mkdir -p _static
make html 2>documentation_errors.txt
num_warnings=$(wc -l < documentation_errors.txt)
echo "Found $num_warnings problem(s) during sphinx-build"
if [[ $num_warnings -gt 1 ]]; then
    echo Sphinx documentation needs to be improved!
    exit 1
fi
echo Sphinx documentation was build successfully
