# CamClient

This client opens a camera feed and does multiple 
calculations with each current frame. 

First, it does a pedestrian detection, 
to ensure that moving humans get detected, 
if they are not very close to the camera. 

If a human gets near the camera, a face detection is 
run as well, where the client compares each face 
in the frame with pictures of humans, 
uploaded to the `profiles` folder.

# Getting started

## Clone the repository
```shell
git clone https://github.com/Cankar001/CamClient.git
```

## Create a environment file
You need to copy the `.env.example` file and rename it to just `.env`.

Fill in the server address and the port, you chose before when intializing the server.

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
# pointing to your env folder
pip --version
```

## Install requirements from requirements.txt
```shell
pip install -r requirements.txt
```

## Run client
```shell
pipenv run python Client.py
```

# Troubleshooting

I discovered, that the virtualenv doesn't seem to like the powershell, 
I couldn't get it working in it.

But using a bash on windows, using `source ./venv/Scripts/activate` worked perfectly fine.
