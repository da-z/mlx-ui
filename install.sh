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

if [ "$1" == "refresh" ]; then

  echo 'refreshing requirements'

  # install deps
  pip3 install \
      mlx-lm \
      streamlit \
      watchdog \

  pip3 freeze > requirements.txt

else

  echo 'installing requirements'

  # install deps
  pip3 install -r requirements.txt

fi

echo 'installation complete'
