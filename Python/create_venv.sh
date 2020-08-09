#!/bin/bash
#
# Copyright © 2020 Johann A. Briffa

python3 -m venv --prompt ASEMI venv

source venv/bin/activate && \
pip install wheel && \
pip install -r requirements.txt && \
pip install -e .
