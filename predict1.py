import os
import face_recognition_api
import pickle
import numpy as np
import pandas as pd
import time


def get_prediction_images(prediction_dir):
    files = [x[2] for x in os.walk(prediction_dir)][0]
    l = []
    exts = [".jpg", ".jpeg", ".png"]
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in exts:
            l.append(os.path.join(prediction_dir, file))

    return l


fname = 'classifier.pkl'
prediction_dir = './test-images'

encoding_file_path = './encoded-images-data.csv'
df = pd.read_csv(encoding_file_path)
full_data = np.array(df.astype(float).values.tolist())

# Extract features and labels
# remove id column (0th column)
X = np.array(full_data[:, 1:-1])
y = np.array(full_data[:, -1:])

if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        (le, clf) = pickle.load(f)
else:
    print('\x1b[0;37;43m' + "Classifier '{}' does not exist".format(fname) + '\x1b[0m')
    quit()

curr = time.time()


for image_path in get_prediction_images(prediction_dir):
    # print colorful text with image name
    print('\x1b[6;30;42m' + "=====Predicting faces in '{}'=====".format(image_path) + '\x1b[0m')

    img = face_recognition_api.load_image_file(image_path)
    faces_encodings = face_recognition_api.face_encodings(img)
    print("Found {} faces in the image".format(len(faces_encodings)))

    for face_encoding in faces_encodings:
        face_encoding = face_encoding.reshape(1, -1)

        predictions = clf.predict_proba(face_encoding).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        print("Predict {} with {:.2f} confidence.".format(person, confidence))

        # print(face_recognition_api.compare_faces(X, face_encoding))
        #
        # predictions = clf.predict(face_encoding).ravel()
        # person = le.inverse_transform(int(predictions[0]))
        # print("Predict {}.".format(person))
    print()

print("Predicted in " + str(time.time() - curr) + " seconds.")

