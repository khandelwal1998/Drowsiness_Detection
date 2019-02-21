# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:06:41 2019

@author: abhishek
"""

import cv2
from scipy.spatial import distance

from imutils import face_utils
import dlib
import imutils
from imutils.face_utils import rect_to_bb

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor('E:\Abhishek\OpenCV\Projects\shape_predictor_68_face_landmarks.dat')

(ll,lr)=face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rl,rr)=face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
while(True):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    subject=detect(gray,0)
    if(subject):
        for i in subject:
            (x,y,w,h)=rect_to_bb(subject[0])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255))
            face=predict(gray,i)
            face = face_utils.shape_to_np(face)
            left_eye=face[ll:lr]
            right_eye=face[rl:rr]
            left_eye_aspect_ration=eye_aspect_ratio(left_eye)
            right_eye_aspect_ration=eye_aspect_ratio(right_eye)
            total_ration=(left_eye_aspect_ration+right_eye_aspect_ration)/2.0
            if(total_ration<0.25):
                flag+=1
                print(flag)
                if(flag>=10):
                    cv2.putText(frame,"***********************Alert open your eyes*********************",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    print("Alert")
            else:
                flag=0
    cv2.imshow("Video",frame)
    if(cv2.waitKey(1) & 0xff==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()