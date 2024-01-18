#!/bin/sh

# remove existing virtual env
if [ -d ".venv" ]; then
    rm -rf .venv
fi

# create virtual env
pip3 install virtualenv
python3 -m virtualenv .venv

# activate virtual env
. .venv/bin/activate

# install deps
pip3 install mlx \
    mlx-lm \
    streamlit \
    autopep8 \
    watchdog \

pip3 freeze > requirements.txt
