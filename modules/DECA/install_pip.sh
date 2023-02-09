#!/usr/bin/env bash

echo "Creating virtual environment"
python3.9 -m venv deca-env
echo "Activating virtual environment"

source $PWD/deca-env/bin/activate
$PWD/deca-env/bin/pip install -r requirements.txt