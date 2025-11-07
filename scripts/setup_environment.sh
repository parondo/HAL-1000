#!/bin/bash

# HALL 1000 Environment Setup Script

echo "Setting up HALL 1000 development environment..."

# Create virtual environment
python3 -m venv hall1000_env
source hall1000_env/bin/activate

# Install core requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 pre-commit

# Install optional dependencies
pip install roboticstoolbox-python spatialmath-python

# Setup pre-commit hooks
pre-commit install

echo "Environment setup complete!"
echo "Activate with: source hall1000_env/bin/activate"
