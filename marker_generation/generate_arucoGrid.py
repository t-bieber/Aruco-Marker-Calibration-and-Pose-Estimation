import cv2
import cv2.aruco as aruco

# Create gridboard, which is a set of Aruco markers
# the following call gets a board of markers 5 wide X 7 tall
gridboard = aruco.GridBoard(
    size=(3,4),
    markerLength=0.1,
    markerSeparation=0.02,
    dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
)

# old syntax
# gridboard = aruco.GridBoard_create(
#         markersX=3, 
#         markersY=4, 
#         markerLength=0.1, 
#         markerSeparation=0.01, 
#         dictionary=aruco.Dictionary_get(aruco.DICT_5X5_50))

# Create an image from the gridboard
img = gridboard.generateImage(outSize=(988, 1400))

# old syntax
# img = gridboard.draw(outSize=(988, 1400))
cv2.imwrite("test_gridboard.jpg", img)

# Display the image to us
cv2.imshow('Gridboard', img)
# Exit on any key
cv2.waitKey(0)
cv2.destroyAllWindows()