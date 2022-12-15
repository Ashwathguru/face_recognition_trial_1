from PIL import ImageGrab
import cv2
import numpy as np
import face_recognition
import os
from win10toast import ToastNotifier
from datetime import datetime
from tkinter import *

#FileDateTime
fileDateFormat = '%Y%m%d%H%M%S'
currentFileDate = datetime.now().strftime(fileDateFormat)

#Windows Notification
toast = ToastNotifier()

def fraud_prompter(msg):
    root = Tk()
    # specify size of window.
    root.geometry("600x150")
    # Create text widget and specify size.
    T = Text(root, height = 5, width = 52)
    # Create label
    alert_msg = "FRAUD DETECTED: \n"+str(msg)
    l = Label(root, text = alert_msg)
    l.config(font =("Courier", 14))
    # Create an Exit button.
    b2 = Button(root, text = "Exit",
            command = root.destroy)
    l.pack()
    b2.pack()
    root.mainloop()

#Read uploaded image, Recognize and encode the face
image = face_recognition.load_image_file(os.path.abspath("images/candidate.jpg"))
image_face_encoding = face_recognition.face_encodings(image)[0]
known_face_encodings = [image_face_encoding]
a = 1
while a == 1:
    screen = np.array(ImageGrab.grab(bbox=(0,0,1920,1080)))

    face_locations = face_recognition.face_locations(screen)
    face_encodings = face_recognition.face_encodings(screen, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.7)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            print('Face matches the input the image !')
        else:
            print('ALERT! : Face does not match the image !')
            fraud_prompter('ALERT! : Face does not match the image !')
            a = 2
            