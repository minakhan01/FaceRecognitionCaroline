import picamera
from time import sleep

camera = picamera.PiCamera()

while True:
	camera.capture('image.jpg')
