import picamera
from time import sleep
import face_recognition_api
import cv2
import os
import pickle
import numpy as np

# Load Face Recogniser classifier
fname = 'classifier.pkl'

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        (le, clf) = pickle.load(f)
else:
    print('\x1b[0;37;43m' + "Classifier '{}' does not exist".format(fname) + '\x1b[0m')
    quit()

camera = picamera.PiCamera()    

input_path = 'image.jpg'
shrink_image = 1

print('before loop')

while True:
    # Grab a single frame of video
    camera.capture(input_path)
    camera.start_preview()
    img = cv2.imread(input_path)
    frame = img
    small_frame = cv2.resize(img, (0, 0), fx=shrink_image, fy=shrink_image)

    # Display the initial image
    #cv2.imshow('FRAME', frame)
    #cv2.waitKey(25)
    print('got initial image')

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition_api.face_locations(small_frame)
        face_encodings = face_recognition_api.face_encodings(small_frame, face_locations)

        face_names = []

        if len(face_locations) > 0:
            print("faces detected")


        # Predict the unknown faces in the video frame
        for face_encoding in face_encodings:
            face_encoding = face_encoding.reshape(1, -1)

            # predictions = clf.predict(face_encoding).ravel()
            # person = le.inverse_transform(int(predictions[0]))

            predictions = clf.predict_proba(face_encoding).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            print(person, confidence)
            if confidence < 0.7:
                person = 'Unknown'

            face_names.append(person.title())
            print('person name:'+person.title())

    #process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= shrink_image
        right *= shrink_image
        bottom *= shrink_image
        left *= shrink_image

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    # Display the resulting image
    #cv2.imshow('FRAME', frame)
    #cv2.waitKey(25)
    camera.stop_preview()

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
