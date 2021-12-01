import cv2
import numpy as np
import os
import faceRecoginisation as fr

# img = cv2.imread(r"C:\Users\dines\Music\Work\20211010_003943.jpg")
cap = cv2.VideoCapture(0)
while True:
    res, img = cap.read()
    facesdetected, Grayimg = fr.faceDetection(img)
    print(facesdetected)
    # for (x, y, w, h) in facesdetected:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # while True:
    #     img = cv2.resize(img, (800, 500))
    #     cv2.imshow("image", img)
    #     if cv2.waitKey(0) & 0xFF == ord("q"):
    #         break
    # cv2.destroyAllWindows()

    # faces, faceIds = fr.label_for_training_data(r"C:\Users\dines\Music\Work\myImgData")
    # face_recognizer = fr.train_classifier(faces, faceIds)
    # face_recognizer.save("training.yml")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(r"C:\Users\dines\Music\Work\opencvprojects\training.yml")
    # fr.drawrect(img, faces)
    name={0:"Dinesh"}
    for face in facesdetected:
        (x, y, w, h) = face
        roi_gray = Grayimg[y:y+h, x:x+h]
        label, confidence = face_recognizer.predict(roi_gray)
        print("label:", label)
        print("confidence: ", confidence)
        fr.drawrect(img, face)
        predicted_name = name[label]
        fr.puttext(img, predicted_name, x, y)
        img = cv2.resize(img, (800, 500))
        cv2.imshow("image", img)
        cv2.waitKey(0)
cv2.destroyAllWindows()
