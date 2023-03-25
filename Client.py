import math
import socket
import threading
from threading import Lock
import time
import cv2

import os
import face_recognition
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np

import pickle
import struct

import Logger
import EnvironmentLoader

usleep = lambda x: time.sleep(x/1000000.0)

envs = EnvironmentLoader.load()
PORT = int(envs['SERVER_PORT'])
IP = envs['SERVER_ADDRESS']
ADDR = (IP, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

threads = []
names = []
pick = []
faceLocations = []

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

def worker(threadLock, rgb, knownFaceEncodings, knownPersons):
    global faceLocations
    global names
    global pick

    threadLock.acquire()
    faceLocations = face_recognition.face_locations(rgb)
    threadLock.release()

    face_encodings = face_recognition.face_encodings(rgb, faceLocations)

    # pedestrian detection
    (rects, weights) = hog.detectMultiScale(rgb, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    threadLock.acquire()
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    threadLock.release()

    name = 'UNKNOWN'
    encoding_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(knownFaceEncodings, encoding)
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matchedIdxs:
                name = knownPersons[i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
            encoding_names.append(name)

    threadLock.acquire()
    names = encoding_names
    threadLock.release()


if __name__ == '__main__':
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    knownPersons = []
    knownFaceEncodings = []

    for file in os.listdir('./profiles/'):
        if file == '.gitkeep':
            continue
            
        try:
            if file.endswith('.png'):
                knownPersons.append(file.replace('.png', ''))
            elif file.endswith('.jpg'):
                knownPersons.append(file.replace('.jpg', ''))

            file = os.path.join('./profiles/', file)
            knownImage = face_recognition.load_image_file(file)
            knownFaceEncodings.append(face_recognition.face_encodings(knownImage)[0])
        except Exception as e:
            # DEBUG
            raise e

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # now open the video capture device
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    join(frame_width, frame_height)

    threadLock = Lock()
    while vid.isOpened():
        try:
            ret, frame = vid.read()
            if not ret:
                break

            start_time = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=250)
            r = frame.shape[1] / float(rgb.shape[1])

            # Create a worker thread that should do all heavy calculations
            current_worker_thread = threading.Thread(target=worker, args=[threadLock, rgb, knownFaceEncodings, knownPersons])
            threads.append(current_worker_thread)
            current_worker_thread.start()

            # wait a small amount to allow the worker thread a head start
            time.sleep(0.0035)

            print(f'Face locations: {faceLocations}')
            print(f'Names: {names}')
            print(f'Picks: {pick}')

            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(faceLocations, names):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            end_time = time.time()
            current_fps = math.ceil(1 / np.round(end_time - start_time, 3))
            Logger.success(f'Current FPS: {current_fps}')

            # Present the frame
            cv2.imshow('Frame', frame)

            # send the image to the server
            send_image_data(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except KeyboardInterrupt as e:
            break

    vid.release()
    cv2.destroyAllWindows()

    anythingDetected = len(names) > 0 or len(pick) > 0
    Logger.success(f'Detected anything: {anythingDetected}')
    leave(anythingDetected)

    for thread in threads:
        thread.join()

    client.close()

    # TODO: The camera should never fail, so add this reboot command at the end, if the program failed for some reason
    #       this would work, because the script would be registered as a startup program and therefore be running
    #       shortly again
    #os.system('sudo shutdown -r now')
