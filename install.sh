#!/bin/sh

# remove existing virtual env
if [ -d venv ]; then
    echo 'recreating virtual env'
    rm -rf venv
fi

pip3 install --upgrade pip

# create virtual env
pip3 install virtualenv
python3 -m virtualenv venv

# activate virtual env
. ./venv/bin/activate

# install deps
pip3 install \
    mlx-lm \
    streamlit \
    watchdog \

pip3 freeze > requirements.txt
