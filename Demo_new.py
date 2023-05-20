import math
import queue
import socket
import threading
import time
import cv2

import os

import pickle
import struct

import Logger
import EnvironmentLoader
import Camera as c

usleep = lambda x: time.sleep(x/1000000.0)

if __name__ == '__main__':
    cam = c.Camera(mirror=True)
    cam.stream()

    # TODO: The camera should never fail, so add this reboot command at the end, if the program failed for some reason
    #       this would work, because the script would be registered as a startup program and therefore be running
    #       shortly again
    #os.system('sudo shutdown -r now')
