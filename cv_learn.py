# READ AN IMAGE
# import cv2 as cv
#
# img = cv.imread('Photos/cat.jpg')
#
# cv.imshow('Cat', img)
#
# cv.waitKey(0)


# READ A VIDEO
# import cv2 as cv
#
# capture = cv.VideoCapture('Videos/dog.mp4')
#
# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video', frame)
#
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
#
# capture.release()
# cv.destroyAllWindows()

# RESIZE VIDEO
# import cv2 as cv
#
# capture = cv.VideoCapture('Videos/dog.mp4')
#
# def rescaleFrame(frame, scale=0.75):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#
#     dimensions = (width, height)
#
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
#
# def changeRes(width, height):
#     capture.set(3, width)
#     capture.set(4, height)
#
# while True:
#     isTrue, frame = capture.read()
#
#     frame_resized = rescaleFrame(frame);
#     cv.imshow('Video', frame)
#     cv.imshow('Video Resized', frame_resized)
#
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
#
# capture.release()
# cv.destroyAllWindows()

# DRAWING A SHAPE
# import cv2 as cv
# import numpy as np
#
# blank = np.zeros((500, 500, 3), dtype='uint8')
# cv.imshow('Blank', blank)

# 1. Point the certain color
# blank[200:300, 300:400] = 0, 0, 255
# cv.imshow('Green', blank)

# 2. Draw a rectangle
# cv.rectangle(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (0, 255, 0), thickness=-1)
# cv.imshow('Rectangle', blank)

# 3. Draw a circle
# cv.circle(blank, (blank.shape[1] // 2, blank.shape[0] // 2), 40, (0, 0, 255), thickness=-1)
# cv.imshow('Circle', blank)

# 4. Draw a line
# cv.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)
# cv.imshow('Line', blank)

# 5. WRITE a text
# cv.putText(blank, 'Hello my name is Khoa Walnut', (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
# cv.imshow('Text', blank)

# cv.waitKey(0)

# CONVERT IMAGE
# import cv2 as cv
#
# img = cv.imread('Photos/park.jpg')
# cv.imshow('Boston', img)
#
# # Convert to grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# # Bluring image
# blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)
#
# # Edge Cascade
# canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny', canny)
#
# # Dilating image
# dilated = cv.dilate(canny, (7, 7), iterations=3)
# cv.imshow('Dilated', dilated)
#
# # Erosion image
# eroded = cv.erode(dilated, (7, 7), iterations=1)
# cv.imshow('Eroded', eroded)
#
# # Resize
# resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
# cv.imshow('Resized', resized)
#
# # Cropping
# cropped = img[50:200, 200:400]
# cv.imshow('Cropped', cropped)
#
# cv.waitKey(0)

# IMAGE TRANSFORMATION
# import cv2 as cv
# import numpy as np
#
# img = cv.imread('Photos/park.jpg')
#
# cv.imshow('Park', img)
#
#
# # Translate
# def translate(img, x, y):
#     transMatrix = np.float32([[1, 0, x], [0, 1, y]])
#     dimensions = (img.shape[1], img.shape[0])
#     return cv.warpAffine(img, transMatrix, dimensions)
#
# # -x left
# # -y up
# # x right
# # y down
#
# translated = translate(img, -100, 100)
# cv.imshow('Translated', translated)
#
# # Rotation
# def rotate(img, angle, rotationPoint = None):
#     (height, width) = img.shape[:2]
#
#     if rotationPoint is None:
#         rotationPoint = (width >> 1, height >> 1)
#
#     rotationMatrix = cv.getRotationMatrix2D(rotationPoint, angle, 1.0)
#     dimensions = (width, height)
#
#     return cv.warpAffine(img, rotationMatrix, dimensions)
#
# rotated = rotate(img, -45)
# cv.imshow('Rotated', rotated)
#
# rotated_rotated = rotate(rotated, -90)
# cv.imshow('Rotated Rotated', rotated_rotated)
#
# # Resizing
# resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
# cv.imshow('Resized', resized)
#
# # Flipping
# flip = cv.flip(img, 1)
# cv.imshow('Flip', flip)
#
# # Cropping
# cropped = img[200:400, 300:400]
# cv.imshow('Cropped', cropped)
#
# cv.waitKey(0)

# CONTOURS USING
# import cv2 as cv
# import numpy as np
#
# img = cv.imread('Photos/cats.jpg')
# cv.imshow('Cats', img)
#
# blank = np.zeros(img.shape, dtype='uint8')
# cv.imshow('Blank', blank)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)
#
# canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny', canny)
#
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)
#
# contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#
# print(f'{len(contours)} contours found')
#
# cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
# cv.imshow('Contours draw', blank)
#
# cv.waitKey(0)

# COLOR SPACES
# import cv2 as cv
# import matplotlib.pyplot as plt
#
# img = cv.imread('Photos/park.jpg')
# cv.imshow('Boston', img)
#
# # plt.imshow(img)
# # plt.show()
#
# # BGR to GRay scaling
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# # BGR to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV', hsv)
#
# # BGR to L * a * b
# lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LAB', lab)
#
# # BGR to RGB
# rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGB', rgb)
#
# # HSV to BGR
# lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
# cv.imshow('LAB => BGR', lab_bgr)
#
# plt.imshow(rgb)
# plt.show()
#
# cv.waitKey(0)

# COLOR CHANNEL
# import cv2 as cv
# import numpy as np
#
# img = cv.imread('Photos/park.jpg')
# cv.imshow('Boston', img)
#
# blank = np.zeros(img.shape[:2], dtype='uint8')
#
# b, g, r = cv.split(img)
#
# blue = cv.merge([b, blank, blank])
# green = cv.merge([blank, g, blank])
# red = cv.merge([blank, blank, r])
#
# cv.imshow('Blue', blue)
# cv.imshow('Green', green)
# cv.imshow('Red', red)
#
# print(img.shape)
# print(b.shape)
# print(g.shape)
# print(r.shape)
#
# merged = cv.merge([b, g, r])
# cv.imshow('Merged', merged)
#
# cv.waitKey(0)

# BLURING IMAGE
# import cv2 as cv
#
# img = cv.imread('Photos/cats.jpg')
# cv.imshow('Cats', img)
#
# # Averaging Blur
# average = cv.blur(img, (3, 3))
# cv.imshow('Average', average)
#
# # Gaussian Blur
# gauss = cv.GaussianBlur(img, (3, 3), 0)
# cv.imshow('Gaussian Blur', gauss)
#
# # Medium Blur
# median = cv.medianBlur(img, 3)
# cv.imshow('Median Blur', median)
#
# # Bilateral
# bilateral = cv.bilateralFilter(img, 10, 35, 25)
# cv.imshow('Bilateral', bilateral)
#
# cv.waitKey(0)

# BITWISE OPERATOR
# import cv2 as cv
# import numpy as np
#
# blank = np.zeros((400, 400), dtype='uint8')
# rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
# circle = cv.circle(blank.copy(), (200, 200), 200, (255, 255, 255), -1)
#
# cv.imshow('Rectangle', rectangle)
# cv.imshow('Circle', circle)
#
# # BITWISE AND => INTERSECT REGIONS
# bitwise_and = cv.bitwise_and(rectangle, circle)
# cv.imshow('Bitwise AND', bitwise_and)
#
# # BITWISE OR => NON INTERSECTING AND INTERSECTING REGIONS
# bitwise_or = cv.bitwise_or(rectangle, circle)
# cv.imshow('Bitwise OR', bitwise_or)
#
# # BITWISE XOR => NOT INTERSECT REGION
# bitwise_xor = cv.bitwise_xor(rectangle, circle)
# cv.imshow('Bitwise XOR', bitwise_xor)
#
# # BITWISE NOT
# bitwise_not = cv.bitwise_not(circle)
# cv.imshow('Bitwise NOT', bitwise_not)
#
# cv.waitKey(0)

# MASKING
# import cv2 as cv
# import numpy as np
#
# img = cv.imread('Photos/cats 2.jpg')
# cv.imshow('Cats', img)
#
# blank = np.zeros(img.shape[:2], dtype='uint8')
# cv.imshow('Blank Image', blank)
#
# circle = cv.circle(blank.copy(), (img.shape[1] // 2 + 45, img.shape[0] // 2), 100, (255, 255, 255), -1)
# # cv.imshow('Mask', circle)
#
# rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
#
# weird_shape = cv.bitwise_and(circle, rectangle)
# cv.imshow('Weird Shape', weird_shape)
#
# masked = cv.bitwise_and(img, img, mask=weird_shape)
# cv.imshow('Masked Image', masked)
#
# cv.waitKey(0)

# HISTOGRAM
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv.imread('Photos/cats.jpg')
# cv.imshow('Cats', img)
#
# blank = np.zeros(img.shape[:2], dtype='uint8')

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# mask = cv.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, (255, 255, 255), -1)
#
# masked = cv.bitwise_and(img, img, mask=mask)
# cv.imshow('Mask', masked)

# GRAYSCALE
# gray_histogram = cv.calcHist([gray], [0], mask, [256], [0, 256])

# GRAY SCALE CHECK HISTOGRAM
# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_histogram)
# plt.xlim([0, 256])
# plt.show()

# COLOR HISTOGRAM
# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# colors = ('b', 'g', 'r')
# for index, column in enumerate(colors):
#     hist = cv.calcHist([img], [index], mask, [256], [0, 256])
#     plt.plot(hist, color=column)
#     plt.xlim([0, 256])
# plt.show()
#
# cv.waitKey(0)

# THRESHOLDING / BINARIZING IMAGE
# import cv2 as cv
#
# img = cv.imread('Photos/cats.jpg')
# cv.imshow('Cats', img)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# # SIMPLE THRESHOLD
# threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# cv.imshow('Simple Threshold', thresh)
#
# # THRESH INV
# threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
# cv.imshow('Simple Threshold Invert', thresh_inv)
#
# # ADAPTIVE THRESHOLDING
# adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 9)
# cv.imshow('Adaptive Threshold', adaptive_thresh)
#
# cv.waitKey(0)

# EDGE DETECTION
# import cv2 as cv
# import numpy as np
#
# img = cv.imread('Photos/park.jpg')
# cv.imshow('Park', img)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
#
# # Laplacian
# lap = cv.Laplacian(gray, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)
#
# # Sobel
# sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0)
# sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1)
# combined_sobel = cv.bitwise_or(sobel_x, sobel_y)
#
# cv.imshow('Sobel X', sobel_x)
# cv.imshow('Sobel Y', sobel_y)
# cv.imshow('Combined Sobel X', combined_sobel)
#
# canny = cv.Canny(gray, 150, 175)
# cv.imshow('Canny', canny)
#
# cv.waitKey(0)

# FACE DETECTION WITH IMAGE
# import cv2 as cv
#
# img = cv.imread('Photos/group 1.jpg')
# cv.imshow('Group Person', img)
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Grayscale Person', gray)
#
# haar_cascade = cv.CascadeClassifier('haar_face.xml')
#
# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
#
# print(f'Number of faces detected: {len(faces_rect)}')
#
# for (x, y, w, h) in faces_rect:
#     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
#
# cv.imshow('Face Detect', img)
#
# cv.waitKey(0)

# FACE DETECTION WITH VIDEO
# import cv2 as cv
#
# capture = cv.VideoCapture(0)
#
# cnt = 1
#
# while True:
#     isTrue, frame = capture.read()
#
#     cnt += 1
#
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#
#     haar_cascade = cv.CascadeClassifier('haar_face.xml')
#
#     faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=17)
#
#     for (x, y, w, h) in faces_rect:
#         cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
#
#     cv.imshow('Video', frame)
#
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
#
# capture.release()
# cv.destroyAllWindows()