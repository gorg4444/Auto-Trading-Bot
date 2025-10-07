#!/bin/bash

echo "Setting up Trading Bot environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Please install Python 3.11+ and try again."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating environment and installing required packages..."
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Launching application..."
python3 main.py

echo "Application closed."
deactivate
