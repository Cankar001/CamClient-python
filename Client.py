import math
import queue
import socket
import threading
import time
import cv2

import mediapipe as mp
import numpy as np

import pickle
import struct

import Logger
import EnvironmentLoader

usleep = lambda x: time.sleep(x/1000000.0)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

connected = False

def send(c, msg):
    """
    Sends a string to the server
    :param msg: The string which should be sent.
    :return:
    """
    message = msg.encode('utf-8')
    c.send(message)


def send_image_data(c, frame):
    """
    Sends a single frame in binary to the server
    :param image:
    :return:
    """

    data = pickle.dumps(frame)
    packed_frame = struct.pack("L", len(data)) + data

    usleep(10000)
    send(c, 'stream')
    usleep(10000)
    c.send(packed_frame)
    usleep(10000)


def join(c, frame_width, frame_height):
    print('Sending join')
    send(c, 'camera_join')
    time.sleep(1)
    send(c, f'{frame_width}x{frame_height}')
    time.sleep(1)


def leave(c, motion_detected):
    print('sending leave')
    send(c, 'camera_leave')

    if motion_detected:
        send(c, 'motion_detected')
    else:
        send(c, 'motion_not_detected')

    time.sleep(1)

def send_to_server(lock: threading.Lock, width, height):
    global sync_queue
    global connected

    envs = EnvironmentLoader.load()
    ip = envs['SERVER_ADDRESS']
    port = int(envs['SERVER_PORT'])

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, port))

    fcount = 0

    lock.acquire()
    connected = True
    lock.release()

    join(client, width, height)

    while connected:
        current_frame = sync_queue.get()
        data = pickle.dumps(current_frame)
        packed_frame = struct.pack("L", len(data)) + data

        print(f'Sending stream... {fcount}')
        usleep(10000)
        send(client, 'stream')

        usleep(10000)
        client.sendall(packed_frame)
        sync_queue.task_done()
        fcount += 1

    leave(client, True)
    client.close()

if __name__ == '__main__':
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    sync_lock = threading.Lock()

    sync_queue = queue.Queue()
    sync_thread = threading.Thread(target=send_to_server, args=[sync_lock, frame_width, frame_height])
    sync_thread.start()

    any_face_detected = False
    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
        while vid.isOpened():
            try:
                ret, frame = vid.read()
                if not ret:
                    connected = False
                    break

                start_time = time.time()

                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.detections:
                    any_face_detected = True
                    for detection in results.detections:
                        mp_drawing.draw_detection(frame, detection)

                end_time = time.time()
                current_fps = math.ceil(1 / np.round(end_time - start_time, 3))
            #    Logger.success(f'Current FPS: {current_fps}')

                # Present the frame
                cv2.imshow('Frame', frame)

                # send the image to the server
                sync_queue.put(frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    connected = False
                    break

            except KeyboardInterrupt as e:
                connected = False
                break

    sync_queue.join()
    sync_thread.join()

    vid.release()
    cv2.destroyAllWindows()

    # TODO: The camera should never fail, so add this reboot command at the end, if the program failed for some reason
    #       this would work, because the script would be registered as a startup program and therefore be running
    #       shortly again
    #os.system('sudo shutdown -r now')
