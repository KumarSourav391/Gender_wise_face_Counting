from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load model
model = load_model('gender_predictor.model')
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# open webcam
webcam = cv2.VideoCapture(0)

classes = ['man', 'woman']

# loop through frames
'''
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
'''

def gender_facecounter(image, m, f, size=0.5):
    ## convert image into gray scaled image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray_image, 1.1,5)
    ## iterating over faces

    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),3)

        cropped_image = np.copy(image[y:y+h,x:x+w])

        ## preprocess the image according to our model
        res_face = cv2.resize(cropped_image, (96,96))
        ## cv2.imshow("cropped image",res_face)
        res_face = res_face.astype("float") / 255.0
        res_face = img_to_array(res_face)
        res_face = np.expand_dims(res_face, axis=0)


        ## model prediction
        result = model.predict(res_face)[0]

        ## get label with max accuracy
        idx = np.argmax(result)
        label = classes[idx]

        ## calculating count
        if label == "woman":
            f = f+1
        else:
            m = m+1

    cv2.rectangle(image,(0,0),(300,30),(255,255,255),-1)
    cv2.putText(image, " females = {},males = {} ".format(f,m),(0,15),
    cv2.FONT_HERSHEY_TRIPLEX,0.6,(255, 101, 125),1)
    cv2.putText(image, " faces detected = " + str(len(faces)),(10,30),
    cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,0),1)

    return image

source = cv2.VideoCapture(0)
''''
while True:
    ret, frame = source.read()
    x = 0
    y = 0
    cv2.imshow("Live Facecount", gender_facecounter(frame, x, y))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

source.release()
cv2.destroyAllWindows()


'''
## loading an image
image = cv2.imread("pic3.jpg") #path to image

## maintaining separate counters
males = 0
females = 0

cv2.imshow("Gender FaceCounter", gender_facecounter(image,males,females ))
cv2.waitKey(0)
cv2.destroyAllWindows()
