#!/bin/bash
#
# Copyright © 2020 Johann A. Briffa

python3 -m venv venv
source venv/bin/activate && pip install -r requirements.txt && pip install -e .
