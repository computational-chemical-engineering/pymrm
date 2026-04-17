# Installation Guide

This guide will help you set up your environment for using PyMRM, a Python package for Multiphase Reactor Modeling. The instructions are tailored for a Windows environment, with recommendations for using Anaconda and VS Code. While VS Code is preferred, instructions for Spyder are also included.

## Prerequisites

1. **Install Python (via Anaconda)**:
   - Download and install [Anaconda](https://www.anaconda.com/products/distribution), which includes Python and many scientific libraries.
   - Ensure you select the option to add Anaconda to your system PATH during installation.

2. **Install VS Code**:
   - Download and install [Visual Studio Code](https://code.visualstudio.com/).
   - Install the Python extension for VS Code from the Extensions Marketplace.

3. **Optional: Install Spyder**:
   - If you prefer Spyder, install it via Anaconda:
     ```sh
     conda install spyder
     ```

## Setting Up a Virtual Environment

It is recommended to use a virtual environment to isolate your project dependencies. You can use either `conda` (preferred with Anaconda) or `venv`.

### Option 1: Using Conda

1. **Create a Conda Environment**:
   - Open a terminal or Anaconda Prompt.
   - Run the following command to create a new environment named `pymrm_env`:
     ```sh
     conda create -n pymrm_env python=3.10
     ```

2. **Activate the Environment**:
   - Run the following command:
     ```sh
     conda activate pymrm_env
     ```

### Option 2: Using venv

1. **Create a Virtual Environment**:
   - Open a terminal.
   - Navigate to your project directory.
   - Run the following command:
     ```sh
     python -m venv .venv
     ```

2. **Activate the Virtual Environment**:
   - On Windows:
     ```sh
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source .venv/bin/activate
     ```

3. **Troubleshooting Windows PowerShell**
   When using Windows PowerShell, it sometimes refuses to execute scripts, and it refuses to run the `activate` script. In that case, run PowerShell as an administrator and execute the command:
   ```sh
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

## Installing PyMRM

PyMRM is available on PyPI and includes all necessary dependencies.

1. **Upgrade pip**:
   - Run the following command to ensure you have the latest version of pip:
     ```sh
     python -m pip install --upgrade pip
     ```

2. **Install PyMRM**:
   - Run the following command:
     ```sh
     pip install pymrm
     ```

## Verifying the Installation

1. **Test the Installation**:
   - Navigate to the `examples` folder in the PyMRM repository or download example notebooks from the [PyMRM documentation](https://computational-chemical-engineering.github.io/pymrm-book).
   - Open a notebook in VS Code or Jupyter Notebook and execute the cells to verify that PyMRM and its dependencies are working correctly.

2. **Optional: Test in Spyder**:
   - Open Spyder and run a Python script that imports PyMRM:
     ```python
     import pymrm
     print("PyMRM version:", pymrm.__version__)
     ```

## Conclusion

You have successfully set up your environment for using PyMRM. Whether you use VS Code or Spyder, you are now ready to start modeling multiphase reactors. For further guidance, refer to the [PyMRM documentation](https://computational-chemical-engineering.github.io/pymrm-book).
