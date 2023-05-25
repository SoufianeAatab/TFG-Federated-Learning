from time import time
import cv2

# Create a new VideoCapture object
cam = cv2.VideoCapture(0)

# Initialise variables to store current time difference as well as previous time call value
previous = time()
delta = 0

max_images = 1000
# Keep looping
for c in range(max_images):
    # Show the image and keep streaming
    _, img = cam.read()
    cv2.imshow("Frame", img)
    img = cv2.resize(img, (256, 256)) 
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("glasses_dataset/yes/img"+str(c)+".jpg", gray_frame)

    cv2.waitKey(1)