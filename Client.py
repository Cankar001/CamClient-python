import socket
import time
import cv2

import os
import face_recognition
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np

import pickle
import struct

usleep = lambda x: time.sleep(x/1000000.0)

PORT = 45457
IP = '127.0.0.1'
ADDR = (IP, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)


def send(msg):
    message = msg.encode('utf-8')
    client.send(message)


def send_image_data(image):
    usleep(10000)
    send('stream')
    usleep(10000)
    client.send(image)
    usleep(10000)


def join(frame_width, frame_height):
    print('Sending join')
    send('join')
    send('camera')

    time.sleep(1)


def leave(motionDetected):
    pass


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
    while vid.isOpened():
        try:
            ret, frame = vid.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=250)
            r = frame.shape[1] / float(rgb.shape[1])

            faceLocations = face_recognition.face_locations(rgb)
            faceEncodings = face_recognition.face_encodings(rgb, faceLocations)

            # pedestrian detection
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            name = 'UNKNOWN'
            for encoding in faceEncodings:
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
                    break

            names.append(name)

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

            cv2.imshow('Frame', frame)

            # send the image to the server
            data = pickle.dumps(frame)
            send_image_data(struct.pack("L", len(data)) + data)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except KeyboardInterrupt as e:
            break

    vid.release()
    cv2.destroyAllWindows()

    # TODO: Send this in the leave process, by this will be decided whether or not to save the video.
    anyFaceDetected = len(names) > 0

    time.sleep(1)
    print('sending leave')
    send('leave')
    client.close()
