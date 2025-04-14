# Setting up a Python Virtual Environment

## What is a Virtual Environment?

A virtual environment is an isolated Python environment that allows you to install packages for a specific project without affecting your system's global Python installation.

## Creating a Virtual Environment

### Option 1: Using venv (Python 3.3+)

The `venv` module is included in the Python standard library.

```bash
# Navigate to your project directory

# Create the virtual environment
python -m venv env

# On some systems, you might need to use python3 explicitly
# python3 -m venv env
```

### Option 2: Using virtualenv (if venv is not available)

```bash
# Install virtualenv if you don't have it
pip install virtualenv

# Create the virtual environment
virtualenv env
```

## Activating the Virtual Environment

### On Linux/macOS:

```bash
source env/bin/activate
```

### On Windows:

```bash
# Command Prompt
env\Scripts\activate.bat

# PowerShell
env\Scripts\Activate.ps1
```

## Installing Packages

After activating the environment, you can install packages with pip:

```bash
pip install package_name

# For your current project, you might want to install:
pip install ib_insync pymongo motor matplotlib numpy scipy pandas pytz
```

## Deactivating the Virtual Environment

When you're done working on the project:

```bash
deactivate
```

## Creating a requirements.txt File

To make your environment reproducible:

```bash
# After installing all necessary packages
pip freeze > requirements.txt

# To install from requirements.txt later
pip install -r requirements.txt
```
