# CamClient

This demo application detects, if someone is in a frame and turns off the connected monitor, if no one was detected for a configurable amount of time. It turns the monitor on again, if someone is detected.

# Getting started

## Clone the repository
```shell
git clone https://github.com/Cankar001/CamClient-python
```

## Create a virtual environment
```shell
python -m venv venv
```

## Activate the virtual enviroment
```shell
# on windows
.\venv\Scripts\activate.bat

# on linux
source ./venv/bin/activate
```

## Check if environment has been activated
```shell
# This command should show the python path,
# pointing to your venv folder
pip --version
```

## Install requirements from requirements.txt
```shell
pip install -r requirements.txt
```

## Run client
```shell
pipenv run python Demo.py
```

# Troubleshooting

I discovered, that the virtualenv doesn't seem to like the powershell, 
I couldn't get it working in it.

But using a bash on windows, using `source ./venv/Scripts/activate` worked perfectly fine.
