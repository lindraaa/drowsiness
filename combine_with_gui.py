import tkinter as tk
from PIL import Image, ImageTk
from imutils.video import VideoStream
from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import cv2
import dlib
import pygame

class DrowsinessDetectorApp:
    def __init__(self, master):
        self.COUNTER = 0
        self.master = master
        self.master.title("Drowsiness Detector")

        self.vs = VideoStream(src=0).start()
        time.sleep(1.0)

        self.alarm_status = False
        self.alarm_status2 = False
        self.saying = False
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 60
        self.YAWN_THRESH = 20

        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        self.frame = tk.Label(self.master)
        self.frame.pack()

        self.quit_button = tk.Button(self.master, text="Quit", command=self.quit)
        self.quit_button.pack()

        self.update_frame()

    def update_frame(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            #eyetrackign
            eye = self.final_ear(shape)
            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            #mouth
            distance = self.lip_distance(shape)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if distance > self.YAWN_THRESH:
                cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not self.alarm_status2 and not self.saying:
                    self.alarm_status2 = True
                    t = Thread(target=self.play_alarm, args=('alarm.wav',))
                    t.daemon = True
                    t.start()
            else:
                self.alarm_status2 = False

            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES and not self.alarm_status:

                    self.alarm_status = True
                    t = Thread(target=self.play_alarm, args=('alarm.wav',))
                    t.daemon = True
                    t.start()
            
            else:
                self.COUNTER = 0
                self.alarm_status = False

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.frame.imgtk = imgtk
        self.frame.config(image=imgtk)
        self.frame.after(10, self.update_frame)

    def final_ear(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        return (ear, leftEye, rightEye)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    def lip_distance(self, shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))

        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))

        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)

        distance = abs(top_mean[1] - low_mean[1])
        return distance

    def play_alarm(self, sound_file):
        pygame.mixer.init()
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()

    def quit(self):
        self.vs.stop()
        self.master.quit()

def main():
    root = tk.Tk()
    app = DrowsinessDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
