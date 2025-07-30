#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install wheel
pip install -r requirements.txt