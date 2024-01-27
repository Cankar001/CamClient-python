import cv2
import time
import os
import datetime
import math
from threading import Thread
from queue import Queue
import imutils

import MonitorControl

class Camera:
    def __init__(self, mirror=False, motion_detection_reference_frame_threshold=1000, motion_detection_threshold=500, seconds_until_monitor_off=300, show_debug_text=False):
        self.data = None
        self.cam = cv2.VideoCapture(0)
        self.show_debug_text = show_debug_text

        self.WIDTH = 640
        self.HEIGHT = 480

        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2
        self.touched_zoom = False

        self.image_queue = Queue()
        self.video_queue = Queue()

        self.scale = 1
        self.__setup()

        self.recording = False

        self.mirror = mirror
        self.first_frame = None
        self.frame_counter = 0

        # if the last X frames were recorded as motion_detected, 
        # reset the reference image.
        self.motion_detection_threshold = motion_detection_threshold
        self.motion_detection_threshold_frame_counter = motion_detection_reference_frame_threshold
        self.motion_detected_counter = 0
        self.motion_detection_reshoot_reference_frame = False
        self.motion_detection_reference_regenerated_counter = 0

        self.monitor_control = MonitorControl.MonitorControl()
        
        self.frames_since_no_motion = 0
        self.seconds_until_monitor_off = seconds_until_monitor_off

    def __setup(self):
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        time.sleep(2)

    def set_location(self, x, y):
        self.center_x = x
        self.center_y = y
        self.touched_zoom = True

    def get_fps(self):
        fps = -1
        (major_ver, _, _) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = self.cam.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = self.cam.get(cv2.CAP_PROP_FPS)
        return fps


    def detect_motion(self, np_image):
        #frame = imutils.resize(np_image, width=500)
        frame = np_image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.first_frame is None or self.motion_detection_reshoot_reference_frame:
            self.first_frame = gray
            self.motion_detection_reshoot_reference_frame = False
            return None
        
        frame_delta = cv2.absdiff(self.first_frame, gray)
        threshold = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        threshold = cv2.dilate(threshold, None, iterations=2)
        cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts

    def drawTextOnImage(self, image, text, coords):
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.5
        font_color = (0, 0, 255)
        thickness = 1
        line_type = 1

        cv2.putText(image, text, coords, font, font_scale, font_color, thickness, line_type)

    def stream(self):
        # streaming thread
        def streaming():
            self.ret = True
            frames_until_monitor_off = 1024
            fps = 60
            while self.ret:
                self.ret, np_image = self.cam.read()
                if np_image is None:
                    continue

                start = time.time()

                if self.mirror:
                    np_image = cv2.flip(np_image, 0)
                    np_image = cv2.flip(np_image, 1)
                if self.touched_zoom:
                    np_image = self.__zoom(np_image, (self.center_x, self.center_y))
                else:
                    if not self.scale == 1:
                        np_image = self.__zoom(np_image)
                
                if np_image is not None:                   
                    # Detect the motion
                    cnts = self.detect_motion(np_image=np_image)
                    if cnts is None:
                        self.frame_counter += 1
                        continue

                    motion_detected = len(cnts) >= 5
                    if motion_detected:
                        self.frames_since_no_motion = 0
                        self.motion_detected_counter += 1
                        # We detected motion, so turn on the monitor again.
                        if not self.monitor_control.isMonitorAwake():
                            self.monitor_control.awakeMonitor()
                    else:
                        self.frames_since_no_motion += 1

                    if self.frames_since_no_motion >= frames_until_monitor_off:
                        print('No motion detected, turning off monitor...')
                        self.monitor_control.shutdownMonitor()
                        self.frames_since_no_motion = 0

                    if self.motion_detected_counter >= self.motion_detection_threshold_frame_counter:
                        print(f'The last {self.motion_detection_threshold_frame_counter} frames were recorded as motion detected, reshooting reference image...')
                        self.motion_detected_counter = 0
                        self.motion_detection_reference_regenerated_counter += 1
                        self.motion_detection_reshoot_reference_frame = True

                        # As we don't have any motion anymore, we can turn off the display
                        if self.monitor_control.isMonitorAwake():
                            self.monitor_control.shutdownMonitor()

                    for c in cnts:
                        if cv2.contourArea(c) < self.motion_detection_threshold:
                            continue

                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(np_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Draw stats
                    if self.show_debug_text:
                        self.drawTextOnImage(np_image, f'Motion detected: {motion_detected}', (0, 20))
                        self.drawTextOnImage(np_image, f'FPS: {fps}', (0, 50))
                        self.drawTextOnImage(np_image, f'Frame counter: {self.frame_counter}', (0, 80))
                        self.drawTextOnImage(np_image, f'Motion detected frame counter: {self.motion_detected_counter} / {self.motion_detection_threshold_frame_counter}', (0, 110))
                        self.drawTextOnImage(np_image, f'Frames since no motion: {self.frames_since_no_motion} / {frames_until_monitor_off}', (0, 140))
                        self.drawTextOnImage(np_image, f'Reference image regenerated count: {self.motion_detection_reference_regenerated_counter}', (0, 170))

                    # Draw frame
                    cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
                    cv2.imshow('Frame', np_image)
                    cv2.setMouseCallback('Frame', self.mouse_callback)

                    # Calculate current fps
                    self.frame_counter += 1
                    end = time.time()
                    frame_time = end - start
                    fps =  math.ceil(1 / frame_time)
                    frames_until_monitor_off = (int)(self.seconds_until_monitor_off * fps)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        # q : close
                        self.release()
                        cv2.destroyAllWindows()
                        break
                    elif key == ord('z'):
                        # z : zoom - in
                        self.zoom_in()
                    elif key == ord('x'):
                        # x : zoom - out
                        self.zoom_out()
                    elif key & 0xFF != 0xFF:
                        # any key was pressed, turn on monitor
                        # and reset the counters, so that the timeout can begin again
                        self.motion_detected_counter = 0
                        self.frames_since_no_motion = 0
                        if not self.monitor_control.isMonitorAwake():
                            self.monitor_control.awakeMonitor()

        Thread(target=streaming).start()

    def __zoom(self, img, center=None):
        height, width = img.shape[:2]
        if center is None:
            center_x = int(width / 2)
            center_y = int(height / 2)
            radius_x, radius_y = int(width / 2), int(height / 2)
        else:
            rate = height / width
            center_x, center_y = center

            if center_x < width * (1-rate):
                center_x = width * (1-rate)
            elif center_x > width * rate:
                center_x = width * rate
            if center_y < height * (1-rate):
                center_y = height * (1-rate)
            elif center_y > height * rate:
                center_y = height * rate

            center_x, center_y = int(center_x), int(center_y)
            left_x, right_x = center_x, int(width - center_x)
            up_y, down_y = int(height - center_y), center_y
            radius_x = min(left_x, right_x)
            radius_y = min(up_y, down_y)

        # zoom
        radius_x, radius_y = int(self.scale * radius_x), int(self.scale * radius_y)

        # size
        min_x, max_x = center_x - radius_x, center_x + radius_x
        min_y, max_y = center_y - radius_y, center_y + radius_y

        cropped = img[min_y:max_y, min_x:max_x]
        new_cropped = cv2.resize(cropped, (width, height))

        return new_cropped

    def touch_init(self):
        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2
        self.touched_zoom = False
        self.scale = 1

    def zoom_out(self):
        # scale zoom-out
        if self.scale < 1:
            self.scale += 0.1
        if self.scale == 1:
            self.center_x = self.WIDTH
            self.center_y = self.HEIGHT
            self.touched_zoom = False

    def zoom_in(self):
        # scale zoom-in
        if self.scale > 0.2:
            self.scale -= 0.1

    def zoom(self, num):
        if num == 0:
            self.zoom_in()
        elif num == 1:
            self.zoom_out()
        elif num == 2:
            self.touch_init()

    def save_picture(self):
        ret, img = self.cam.read()
        if ret:
            now = datetime.datetime.now()
            date = now.strftime('%Y%m%d')
            hour = now.strftime('%H%M%S')
            user_id = '00001'
            filename = './images/cvui_{}_{}_{}.png'.format(date, hour, user_id)
            cv2.imwrite(filename, img)
            self.image_queue.put_nowait(filename)

    def record_video(self):
        fc = 20.0
        record_start_time = time.time()
        now = datetime.datetime.now()
        date = now.strftime('%Y%m%d')
        t = now.strftime('%H')
        num = 1
        filename = 'videos/cvui_{}_{}_{}.avi'.format(date, t, num)
        while os.path.exists(filename):
            num += 1
            filename = 'videos/cvui_{}_{}_{}.avi'.format(date, t, num)
        codec = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter(filename, codec, fc, (int(self.cam.get(3)), int(self.cam.get(4))))
        while self.recording:
            if time.time() - record_start_time >= 600:
                self.record_video()
                break
            ret, frame = self.cam.read()
            if ret:
                if len(os.listdir('./videos')) >= 100:
                    name = self.video_queue.get()
                    if os.path.exists(name):
                        os.remove(name)
                out.write(frame)
                self.video_queue.put_nowait(filename)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

    def show(self):
        while True:
            frame = self.data
            if frame is not None:
                cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
                cv2.imshow('Frame', frame)
                cv2.setMouseCallback('Frame', self.mouse_callback)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                # q : close
                self.release()
                cv2.destroyAllWindows()
                break
            elif key == ord('z'):
                # z : zoom - in
                self.zoom_in()
            elif key == ord('x'):
                # x : zoom - out
                self.zoom_out()
            elif key == ord('p'):
                # p : take picture and save image (image folder)
                self.save_picture()
            elif key == ord('v'):
                # v : zoom
                self.touch_init()
            elif key == ord('r'):
                # r : record video
                self.recording = not self.recording
                if self.recording:
                    t = Thread(target=self.record_video)
                    t.start()

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.set_location(x, y)
            self.zoom_in()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.zoom_out()
