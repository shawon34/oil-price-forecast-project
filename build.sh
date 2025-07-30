#!/usr/bin/env bash
set -o errexit

# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# Install Python dependencies
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt