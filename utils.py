import cv2
import numpy as np
from tensorflow.keras.models import load_model


def pre_processing(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (15, 15), 1)

    # threshold makes it either a 1 or 0 (white or black) based on certain parameters
    # see documentation for parameter info
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 3, 5)
    return img_gray, img_blur, img_threshold


def stack_images(imgArray, scale, labels=[], label_text_size=0.7):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, labels[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, label_text_size, (255, 0, 255), 2)
    return ver


def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    # for each contour in contours
    for c in contours:
        area = cv2.contourArea(c)
        if area > 50:
            perimeter = cv2.arcLength(c, True)
            # polygon
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if area > max_area and len(approx == 4):
                biggest = approx
                max_area = area

    return biggest, max_area


# orders a square's points properly for the transform
def reorder(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]

    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]

    return my_points_new


def split_boxes(img, number):
    # splitting image vertically into 9 rows
    rows = np.vsplit(img, number)
    boxes = []
    for row in rows:
        columns = np.hsplit(row, number)
        for box in columns:
            boxes.append(box)
    return boxes


def initialize_prediction_model(path):
    model = load_model(path)
    return model


def get_prediction(boxes, model):
    result = []

    for image in boxes:
        # preprocessing each box's image
        img = np.asarray(image)
        #img = img[4:img.shape[0] - 4, 4:image[1] - 4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)

        # get prediction
        predictions = model.predict(img)

        class_index = np.argmax(predictions, axis=-1)
        probability_value = np.amax(predictions)

        # save to result
        if probability_value > 0.8:
            result.append(class_index[0])
        else:
            result.append(0)

    return result


def display_numbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0, 9):
        for y in range(0, 9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]), (x*secW+int(secW/2) - 10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)

    return img