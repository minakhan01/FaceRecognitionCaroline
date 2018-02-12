import picamera
from time import sleep
import face_recognition_api
import cv2
import os
import pickle
import numpy as np
from PIL import Image

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True



camera = picamera.PiCamera()    

input_path = 'image.jpg'
shrink_image = 1

print('before loop')

while True:
    # Grab a single frame of video
    # camera.start_preview()
    camera.capture(input_path)
    # camera.stop_preview()
    img = cv2.imread(input_path)
    im = Image.open('image.jpg')
    im.show()
