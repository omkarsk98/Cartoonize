# python3 Cartoonize.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/jack.jpg
"""
    jaw (0, 17)
    right_eyebrow (17, 22)
    left_eyebrow (22, 27)
    nose (27, 36)
    right_eye (36, 42)
    left_eye (42, 48)
    mouth (48, 68)
"""
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from imutils import face_utils  # for image processing
import numpy as np
import argparse  # for parsing the command line arguements
import imutils
import dlib  # for documentation
import cv2  # opencv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# parse the command line arguements
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it
image = mpimg.imread(args["image"])
image = imutils.resize(image, width=500)

# detect faces in the original image
rects = detector(image, 1)
# loop over the face detections
for (i, rect) in enumerate(rects):
    """ 
        shape array contains the list of coordinates that have the facial features.
        Indices are as above mentioned
    """
    shape = predictor(image, rect)
    shape = face_utils.shape_to_np(shape)

    eye_brow_color = (0, 0, 0)

    # loop through nose
    # however, the color of the nose would be same as of the mouth
    nose_x, nose_y, nose_z = [], [], []
    x_coordinates_for_mouth, y_coordinates_for_mouth = [], []
    for (x, y) in shape[27:36]:
        nose_x.append(image[x, y][0])
        nose_y.append(image[x, y][1])
        nose_z.append(image[x, y][2])
    mean_x = np.median(np.array(nose_x, dtype=int))
    mean_y = np.median(np.array(nose_y, dtype=int))
    mean_z = np.median(np.array(nose_z, dtype=int))
    nose = (mean_x, mean_y, mean_z)

    mouth = (164, 90, 82)
    nose = mouth
    jaw = (0, 0, 0)
    eye = (255/2, 255/2, 255/2)

    # cover the eyebrows with the color of the skin
    x = image[int((shape[21][0]+shape[22][0])/2), shape[21][1]][0]
    y = image[int((shape[21][0]+shape[22][0])/2), shape[21][1]][1]
    z = image[int((shape[21][0]+shape[22][0])/2), shape[21][1]][2]
    eye_brow_color = (float(x), float(y), float(z))

    # # set the color of the nose using the thresholds
    # x = image[int((shape[21][0]+shape[22][0])/2), shape[33][1]][0]
    # y = image[int((shape[21][0]+shape[22][0])/2), shape[33][1]][1]
    # z = image[int((shape[21][0]+shape[22][0])/2), shape[33][1]][2]
    # if x > 150:
    #     x = 150
    # if y > 150:
    #     y = 120
    # if z > 150:
    #     z = 120
    # nose = (float(x), float(y), float(z))
    # eye_brow_color = (255, 231, 204)
    # visualize all facial landmarks with a transparent overlay
    output = face_utils.visualize_facial_landmarks(image, shape, [
                                                   mouth, eye_brow_color, eye_brow_color, eye, eye, nose, jaw])

thickness = 10
# nose = (254,214,198)
midShift = 20

# """ Animate nose """
# # # points = np.array([[shape[27][0],shape[27][1]],[shape[29][0]-midShift,shape[29][1]],[shape[31][0],shape[31][1]],[shape[33][0],shape[33][1]],[shape[35][0],shape[35][1]],[shape[29][0]+midShift,shape[29][1]]])
# one = [shape[27][0], shape[27][1]]
# two = [shape[31][0]+5, shape[31][1]]
# three = [shape[31][0], shape[32][1]]
# four = [shape[32][0]-5, shape[32][1]]
# five = [shape[33][0], shape[33][1]+5]
# six = [shape[34][0]+5, shape[34][1]]
# seven = [shape[35][0], shape[34][1]]
# eight = [shape[35][0]-5, shape[35][1]]
# # points = np.array([one,two,three,four,five,six,seven,eight])
# points = np.array([one, two, five, eight])
# output = cv2.fillConvexPoly(output, points, eye_brow_color)

""" Animate left eye """
eyes = (230, 230, 230)
one = [shape[36][0]-10, shape[36][1]]
two = [shape[37][0], shape[37][1]-10]
three = [shape[38][0], shape[38][1]-10]
four = [shape[39][0]+10, shape[39][1]]
five = [shape[40][0], shape[40][1]+10]
six = [shape[41][0], shape[41][1]+10]
points = np.array([one, two, three, four, five, six])
output = cv2.fillConvexPoly(output, points, eyes)
""" put iris within the eye ball """
iris_x = int((shape[38][0]+shape[41][0])/2)
iris_y = int((shape[38][1]+shape[41][1])/2)
output = cv2.circle(output, (iris_x, iris_y), 10, (0, 0, 0), -1)

""" Animate right eye """
one = [shape[42][0]-10, shape[42][1]]
two = [shape[43][0], shape[43][1]-10]
three = [shape[44][0], shape[44][1]-10]
four = [shape[45][0]+10, shape[45][1]]
five = [shape[46][0], shape[46][1]+10]
six = [shape[47][0], shape[47][1]+10]
points = np.array([one, two, three, four, five, six])
output = cv2.fillConvexPoly(output, points, eyes)
""" put iris within the eye ball """
iris_x = int((shape[44][0]+shape[47][0])/2)
iris_y = int((shape[44][1]+shape[47][1])/2)
output = cv2.circle(output, (iris_x, iris_y), 10, (0, 0, 0), -1)

""" Animate Left Eyebrow """
left_one_x = int((shape[21][0]+shape[27][0])/2)
left_one_y = int((shape[21][1]+shape[27][1])/2)
left_two_x = int(shape[21][0])
left_two_y = int(shape[21][1])-5
left_three_x = int(shape[18][0])
left_three_y = int(shape[17][1]-15)
# two = int((shape[44][1]+shape[47][0])/2)
output = cv2.line(output, (left_one_x, left_one_y),
                  (left_two_x, left_two_y), (0, 0, 0), 5)
output = cv2.line(output, (left_two_x, left_two_y),
                  (left_three_x, left_three_y), (0, 0, 0), 5)

""" Animate Right Eyebrow """
right_one_x = int((shape[22][0]+shape[27][0])/2)
right_one_y = int((shape[22][1]+shape[27][1])/2)
right_two_x = int(shape[22][0])
right_two_y = int(shape[22][1])-5
right_three_x = int(shape[25][0])
right_three_y = int(shape[25][1]-5)
# two = int((shape[44][1]+shape[47][0])/2)
output = cv2.line(output, (right_one_x, right_one_y),
                  (right_two_x, right_two_y), (0, 0, 0), 5)
output = cv2.line(output, (right_two_x, right_two_y),
                  (right_three_x, right_three_y), (0, 0, 0), 5)


#  Smoothen the image so as to remove the sharp edges
for i in range(20):
    result = ndimage.uniform_filter(output, size=(5, 5, 1))

# Image Dilation to remove random nose
kernel = np.ones((6, 6), np.uint8)
diluted = cv2.dilate(result, kernel, iterations=1)

# # darkened
# darkened = np.double(cartoon)
# darkened = darkened * 0.8
# darkened = np.uint8(darkened)

# Image Erosion to restructure the face with a smoothened view
kernel = np.ones((5, 5), np.uint8)
erode = cv2.erode(diluted, kernel, iterations=1)

""" 
# This code isnt required at all. It disturbs the image
# # opening
# kernel = np.ones((5,5), np.uint8)
# opened = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)

# # closing
# closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

# # Morphological Gradient
# gradient = cv2.morphologyEx(opened, cv2.MORPH_GRADIENT, kernel)

# # adding gradient to eroded image
# gradient_subtracted = erode - gradient

# # gray scale of erored image
# gray = cv2.cvtColor(erode, cv2.COLOR_BGR2GRAY)

# # detecting edges
# edges = cv2.Canny(erode,100,200,200)

# # highlight edges
# # highlighted_edges = """


# Cartoonize
gray = cv2.cvtColor(erode, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
edges = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
color = cv2.bilateralFilter(erode, 9, 300, 300)
cartoon = cv2.bitwise_and(color, color, mask=edges)

# show the image
x_plot, y_plot = 1, 2
fig = plt.figure()
fig.add_subplot(x_plot, y_plot, 1)
plt.imshow(image)
plt.axis("off")
plt.title("Original")
# fig.add_subplot(x_plot, y_plot, 2)
# plt.imshow(output)
# plt.axis("off")
# plt.title("Features highlighted")
# fig.add_subplot(x_plot, y_plot, 3)
# plt.imshow(result)
# plt.axis("off")
# plt.title("smoothened")
# fig.add_subplot(x_plot, y_plot, 4)
# plt.imshow(diluted)
# plt.axis("off")
# plt.title("Dilation")
# fig.add_subplot(x_plot, y_plot, 5)
# plt.imshow(erode)
# plt.axis("off")
# plt.title("Eroded")
fig.add_subplot(x_plot, y_plot, 2)
plt.imshow(cartoon)
plt.axis("off")
plt.title("Cartoonized")

""" 
# This code isnt required because the images are more disturbed
# fig.add_subplot(x_plot, y_plot, 6)
# plt.imshow(opened)
# plt.axis("off")
# plt.title("Opening")
# fig.add_subplot(x_plot, y_plot, 7)
# plt.imshow(closed)
# plt.axis("off")
# plt.title("Closing")
# fig.add_subplot(x_plot, y_plot, 8)
# plt.imshow(gradient)
# plt.axis("off")
# plt.title("Gradient")
# fig.add_subplot(x_plot, y_plot, 9)
# plt.imshow(gradient_subtracted)
# plt.axis("off")
# plt.title("Gradient - Eroded")
# fig.add_subplot(x_plot, y_plot, 10)
# plt.imshow(gray)
# plt.axis("off")
# plt.title("Gray Scale")
# fig.add_subplot(x_plot, y_plot, 11)
# plt.imshow(edges)
# plt.axis("off")
# plt.title("Edges")
# fig.add_subplot(3, 2, 6)
# plt.imshow(cartoon)
# plt.axis("off")
# plt.title("Cartoonized") """


figure_to_save = plt.gcf()
plt.show()
plt.draw()
figure_to_save.savefig('Cartoon.jpg', bbox_inches='tight')
