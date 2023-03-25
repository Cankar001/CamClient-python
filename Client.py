import math
import socket
import time
import cv2

import os
import face_recognition
import mediapipe as mp
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np

import pickle
import struct

import Logger
import EnvironmentLoader

usleep = lambda x: time.sleep(x/1000000.0)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

envs = EnvironmentLoader.load()
PORT = int(envs['SERVER_PORT'])
IP = envs['SERVER_ADDRESS']
ADDR = (IP, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)


def send(msg):
    """
    Sends a string to the server
    :param msg: The string which should be sent.
    :return:
    """
    message = msg.encode('utf-8')
    client.send(message)


def send_image_data(frame):
    """
    Sends a single frame in binary to the server
    :param image:
    :return:
    """

    data = pickle.dumps(frame)
    packed_frame = struct.pack("L", len(data)) + data

    usleep(10000)
    send('stream')
    usleep(10000)
    client.send(packed_frame)
    usleep(10000)


def join(frame_width, frame_height):
    print('Sending join')
    send('camera_join')
    time.sleep(1)
    send(f'{frame_width}x{frame_height}')
    time.sleep(1)


def leave(motion_detected):
    print('sending leave')
    send('camera_leave')

    if motion_detected:
        send('motion_detected')
    else:
        send('motion_not_detected')

    time.sleep(1)


if __name__ == '__main__':
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    knownPersons = []
    knownImages = []
    knownFaceEncodings = []

    for file in os.listdir('./profiles/'):
        if file == '.gitkeep':
            continue
            
        try:
            knownPersons.append(file.replace('.png', ''))
            file = os.path.join('./profiles/', file)
            knownImages = face_recognition.load_image_file(file)
            knownFaceEncodings.append(face_recognition.face_encodings(knownImages)[0])
        except Exception as e:
            # DEBUG
            raise e

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # now read a test video with opencv and send the bytes directly
    #vid = cv2.VideoCapture('test.mp4')
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    join(frame_width, frame_height)

    names = []

    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
        while vid.isOpened():
            try:
                ret, frame = vid.read()
                if not ret:
                    break

                start_time = time.time()

                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(frame, detection)

                end_time = time.time()
                current_fps = math.ceil(1 / np.round(end_time - start_time, 3))
                Logger.success(f'Current FPS: {current_fps}')

                cv2.imshow('Frame', frame)

                # send the image to the server
                #send_image_data(frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except KeyboardInterrupt as e:
                break

    vid.release()
    cv2.destroyAllWindows()

    # TODO: Send this in the leave process, by this will be decided whether or not to save the video.
    anyFaceDetected = len(names) > 0
    leave(anyFaceDetected)
    client.close()

    # TODO: The camera should never fail, so add this reboot command at the end, if the program failed for some reason
    #       this would work, because the script would be registered as a startup program and therefore be running
    #       shortly again
    #os.system('sudo shutdown -r now')
