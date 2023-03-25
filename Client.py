import math
import queue
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

import Logger
import EnvironmentLoader
from multiprocessing import Pool, Queue

usleep = lambda x: time.sleep(x/1000000.0)

def send(c: socket.socket, msg: str):
    """
    Sends a string to the server
    :param c: The client connection handle.
    :param msg: The string which should be sent.
    :return:
    """
    message = msg.encode('utf-8')
    c.send(message)


def send_image_data(c: socket.socket, image):
    """
    Sends a single frame in binary to the server
    :param c: The client connection handle.
    :param image: The frame to send.
    :return:
    """

    data = pickle.dumps(image)
    packed_frame = struct.pack("L", len(data)) + data

    usleep(10000)
    send(c, 'stream')
    usleep(10000)
    c.sendall(packed_frame)
    usleep(10000)


def join(c: socket.socket, width: int, height: int):
    """
    Sends a join request to the server, as well as the width and the height of the connected camera.
    :param c: The client connection handle.
    :param width: The width of the opened camera stream.
    :param height: The height of the opened camera stream.
    :return:
    """
    Logger.info('Sending join')
    send(c, 'camera_join')
    time.sleep(1)
    send(c, f'{width}x{height}')
    time.sleep(1)


def leave(c, motion_detected):
    """
    Sends a leave request to the server, as well as the information whether humans were detected during runtime.
    :param c: The client connection handle.
    :param motion_detected: A flag, indicating if any motion was found during runtime.
    :return:
    """
    Logger.info('sending leave')
    send(c, 'camera_leave')

    if motion_detected:
        send(c, 'motion_detected')
    else:
        send(c, 'motion_not_detected')

    time.sleep(1)

def init_pool(db, rb):
    global detection_buffer
    global render_buffer
    detection_buffer = db
    render_buffer = rb

def detect_object(render_frame, image, knownFaceEncodings, knownPersons, aspect):
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # pedestrian detection
    (rects, weights) = hog.detectMultiScale(render_frame, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

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

    result = {
        'render_frame': render_frame,
        'names': encoding_names,
        'pick': pick,
        'aspect': aspect,
        'face_locations': face_locations
    }

    Logger.success('Worker task was successful')
    detection_buffer.put(result)

def show():
    while True:
        data = detection_buffer.get()
        if data is None:
            break

        Logger.success('Received frame from worker...')

        current_render_frame = data['render_frame']
        current_names = data['names']
        current_pick = data['pick']
        current_aspect = data['aspect']
        current_face_locations = data['face_locations']

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(current_face_locations, current_names):
            # rescale the face coordinates
            top = int(top * current_aspect)
            right = int(right * current_aspect)
            bottom = int(bottom * current_aspect)
            left = int(left * current_aspect)

            # draw the predicted face name on the image
            cv2.rectangle(current_render_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(current_render_frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in current_pick:
            cv2.rectangle(current_render_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        print(f'Face locations: {current_face_locations}')
        print(f'Names: {current_names}')
        print(f'Picks: {current_pick}')

        detected = len(current_names) > 0 or len(current_pick) > 0

        result = {
            'detected': detected,
            'frame': current_render_frame,
        }

        Logger.success('Frame successfully post-processed')
        detection_buffer.task_done()
        render_buffer.put(result)


if __name__ == '__main__':
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    envs = EnvironmentLoader.load()
    port = int(envs['SERVER_PORT'])
    ip = envs['SERVER_ADDRESS']

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, port))

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

    # now open the video capture device
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    join(client, frame_width, frame_height)

    # Create a worker thread that should do all heavy calculations
    detection_buffer = Queue()
    render_buffer = Queue()

    # 6 workers: 1 for the show task and 5 to process frames:
    pool = Pool(6, initializer=init_pool, initargs=(detection_buffer, render_buffer))
    # run the "show" task:
    show_process = pool.apply_async(show, args=())

    frame_workers = []
    anything_detected = False
    last_frame_for_server = None
    while vid.isOpened():
        try:
            success, frame = vid.read()
            if not success:
                break

            anything_detected = False
            rgb = imutils.resize(frame, width=250)
            aspect_ratio = frame.shape[1] / float(rgb.shape[1])

            f = pool.apply_async(detect_object, args=[frame, rgb, knownFaceEncodings, knownPersons, aspect_ratio])
            frame_workers.append(f)

            try:
                render_data = render_buffer.get(block=False)
                anything_detected = render_data['detected']
                frame_for_server = render_data['frame']
                last_frame_for_server = frame_for_server

                # send the image to the server
                send_image_data(client, frame_for_server)

                render_buffer.task_done()
            except queue.Empty:
                anything_detected = False
                if last_frame_for_server is not None:
                    send_image_data(client, last_frame_for_server)

            # Present the frame
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except KeyboardInterrupt as e:
            break

    vid.release()
    cv2.destroyAllWindows()

    # wait for all the frame-putting tasks to complete:
    for f in frame_workers:
        f.get()

    # signal the "show" task to end by placing None in the queue
    detection_buffer.put(None)
    show_process.get()

    Logger.success(f'Detected anything: {anything_detected}')
    leave(client, True)

    client.close()

    # TODO: The camera should never fail, so add this reboot command at the end, if the program failed for some reason
    #       this would work, because the script would be registered as a startup program and therefore be running
    #       shortly again
    #os.system('sudo shutdown -r now')
