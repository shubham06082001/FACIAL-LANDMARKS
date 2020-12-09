import numpy as np
import cv2
import dlib

webcam = True

cap = cv2.VideoCapture(0)

img = cv2.imread('red.jpg')
# img = cv2.resize(img, (0, 0), None, 0.1, 0.1)

img_original = img.copy()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def empty(a):
    pass


cv2.namedWindow("BGR")
cv2.resizeWindow("BGR", 640, 240)
cv2.createTrackbar("Blue", "BGR", 0, 255, empty)
cv2.createTrackbar("Green", "BGR", 0, 255, empty)
cv2.createTrackbar("Red ", "BGR", 0, 255, empty)


def create_box(img, points, scale=5, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)
        # cv2.imshow('Mask', img)

    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        img_crop = img[y:y + h, x:x + w]
        img_crop = cv2.resize(img_crop, (0, 0), None, scale, 3)
        return img_crop

    else:
        return mask


while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread('lena.png')
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    img_original = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        # img_original = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(img_gray, face)
        my_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            my_points.append([x, y])
            # cv2.circle(img_original, (x, y), 5, (50, 50, 255), cv2.FILLED)
            # cv2.putText(img_original, str(n), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 1)

        my_points = np.array(my_points)
        # img_left_eye = create_box(img, my_points[36: 42])
        img_lips = create_box(img, my_points[48:61], 3, masked=True, cropped=False)

        img_color_lips = np.zeros_like(img_lips)
        b = cv2.getTrackbarPos('Blue', 'BGR')
        g = cv2.getTrackbarPos('Green', 'BGR')
        r = cv2.getTrackbarPos('Red', 'BGR')
        # actual without trackbar
        # img_color_lips[:] = 153, 0, 157

        # with track bar
        img_color_lips[:] = b, g, r
        img_color_lips = cv2.bitwise_and(img_lips, img_color_lips)
        img_color_lips = cv2.GaussianBlur(img_color_lips, (7, 7), 10)
        # Image with lips coloured only
        img_color_lips = cv2.addWeighted(img_original, 1, img_color_lips, 0.4, 0)
        # cv2.imshow("COLORED", img_color_lips)
        # image with lips coloured and black & white image
        img_original_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        img_original_gray = cv2.cvtColor(img_original_gray, cv2.COLOR_GRAY2BGR)
        img_color_lips = cv2.addWeighted(img_original_gray, 1, img_color_lips, 0.4, 0)
        cv2.imshow("BGR", img_color_lips)

        # cv2.imshow("lips", img_lips)
        print(my_points)

    cv2.imshow("Original Image", img_original)

    cv2.waitKey(1)
