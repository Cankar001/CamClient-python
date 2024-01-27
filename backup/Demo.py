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

usleep = lambda x: time.sleep(x/1000000.0)

if __name__ == '__main__':
    # now read a test video with opencv and send the bytes directly
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

    scale = 1

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    while vid.isOpened():
        try:
            ret, frame = vid.read()
            if not ret:
                break
     
            frame = cv2.flip(frame, 0)
            
            current_height, current_width, current_channels = frame.shape
            radius_x, radius_y = int(scale * current_height / 100), int(scale * current_width / 100)
            
            center_x, center_y = int(current_height / 2), int(current_width / 2)
            min_x, max_x = center_x - radius_x, center_x + radius_x
            min_y, max_y = center_y - radius_y, center_y + radius_y
                
            cropped = frame[min_x:max_x , min_y:max_y]
            resized_cropped = cv2.resize(cropped, (current_width, current_height))
                
            cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
            cv2.imshow('Frame', resized_cropped)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except KeyboardInterrupt as e:
            break

    vid.release()
    cv2.destroyAllWindows()

    # TODO: The camera should never fail, so add this reboot command at the end, if the program failed for some reason
    #       this would work, because the script would be registered as a startup program and therefore be running
    #       shortly again
    #os.system('sudo shutdown -r now')
