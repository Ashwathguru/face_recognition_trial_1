from PIL import ImageGrab
import cv2
import numpy as np
import face_recognition
import os
from win10toast import ToastNotifier
from datetime import datetime
#FileDateTime
fileDateFormat = '%Y%m%d%H%M%S'
currentFileDate = datetime.now().strftime(fileDateFormat)

#Windows Notification
toast = ToastNotifier()

#Take Screenshot, Save_Image, Recognize and encode the face
screen = np.array(ImageGrab.grab(bbox=(0,0,1920,1080)))
filename = 'images/screenshot_'+currentFileDate+'.png'
cv2.imwrite(filename, screen)
face_locations = face_recognition.face_locations(screen)
face_encodings = face_recognition.face_encodings(screen, face_locations)

#Read uploaded image, Recognize and encode the face
image = face_recognition.load_image_file(os.path.abspath("images/candidate.jpg"))
image_face_encoding = face_recognition.face_encodings(image)[0]

known_face_encodings = [image_face_encoding]

def face_recog():
    if len(face_encodings) > 0:
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.7)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                print('Face matches the input the image !')
                toast.show_toast("E-yes",
                                "Face matches the input the image !",
                                duration = 20,
                                icon_path = "icon/E-yes_icon.ico",
                                threaded = True)
                break
            else:
                print('ALERT! : Face does not match the image !')
                toast.show_toast("E-yes",
                                'ALERT! : Face does not match the image !',
                                duration = 20,
                                icon_path = "icon/E-yes_icon.ico",
                                threaded = True)
                break
        return 1
    else:
        toast.show_toast("E-yes",
                         'No Face Detected !',
                         duration = 20,
                         icon_path = "icon/E-yes_icon.ico",
                         threaded = True)
        return 0

face_recog()