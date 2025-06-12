#!/bin/sh

# remove existing virtual env
if [ -d venv ]; then
    echo '⏳ Recreating virtual env..'
    rm -rf venv
fi

# create virtual env
pipx install virtualenv
virtualenv venv

# activate virtual env
. ./venv/bin/activate

if [ "$1" = "refresh" ]; then

  echo '⏳ Refreshing requirements..'

  # install deps
  pip install \
      mlx-lm \
      tiktoken \
      sentencepiece \
      streamlit \
      watchdog \

  pip freeze > requirements.txt

else

  # install deps
  echo '⏳ Installing requirements..'
  pip install -r requirements.txt

fi

echo '✅ Installation complete. You can use ./run.sh to launch the app'
