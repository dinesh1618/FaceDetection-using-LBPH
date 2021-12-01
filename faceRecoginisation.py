import cv2
import numpy as np
import os

def faceDetection(testImage):
    Grayimg = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    face_haar = cv2.CascadeClassifier(r"C:\Users\dines\Music\Work\haaar\haarcascade_frontalface_default.xml")
    faces= face_haar.detectMultiScale(Grayimg, scaleFactor=1.32, minNeighbors=5)
    return faces, Grayimg

def label_for_training_data(directory):
    faces = []
    faceIds = []
    for path, subdir, files in os.walk(directory):
        for file in files:
            if file.startswith("."):
                continue
            id = os.path.basename(path)
            imgpath = os.path.join(path, file)
            print("id: ",id)
            print("img: ", imgpath)
            test_img = cv2.imread(imgpath)
            faces_rect, GrayImg = faceDetection(test_img)
            print(len(faces_rect))
            if len(faces_rect)!=1:
                continue
            (x, y, w, h) = faces_rect[0]
            roiGray = GrayImg[y:y+w, x:x+h]
            faces.append(roiGray)
            faceIds.append(int(id))
    return faces, faceIds

def train_classifier(faces, faceIds):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceIds))
    return face_recognizer

def drawrect(img, faces):
    (x, y, w, h) = faces
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

def puttext(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 0, 0), 6)
