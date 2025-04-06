import numpy as np
import cv2
import cv2.aruco as aruco

# Select type of aruco marker (size)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Create an image from the marker
# second param is ID number
# last param is total image size
img = aruco.generateImageMarker(aruco_dict, 0, 200)
cv2.imwrite("test_marker.jpg", img)

# # Display the image to us
cv2.imshow('test_marker.jpg', img)

# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()