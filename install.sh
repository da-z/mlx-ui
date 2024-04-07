#!/bin/sh

# remove existing virtual env
if [ -d venv ]; then
    echo '⏳ Recreating virtual env..'
    rm -rf venv
fi

# create virtual env
python3 -m pip install virtualenv
python3 -m virtualenv venv

# activate virtual env
. ./venv/bin/activate

if [ "$1" = "refresh" ]; then

  echo '⏳ Refreshing requirements..'

  # install deps
  pip install \
      mlx-lm \
      tiktoken \
      streamlit \
      watchdog \

  pip freeze > requirements.txt

else

  # install deps
  echo '⏳ Installing requirements..'
  pip install -r requirements.txt

fi

echo '✅ Installation complete. You can use ./run.sh to launch the app'
