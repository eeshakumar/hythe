#!/bin/bash
virtualenv -p /usr/bin/python3.7 ./python/venv --system-site-packages
source ./python/venv/bin/activate && pip install -r requirements.txt