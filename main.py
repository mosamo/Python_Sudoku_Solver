# this prevents warning from showing up
print("Setting up..")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from utils import *
from solve_algorthim import *

# SUDOKU SOLVER
# (if no change then soduko is unsolvable, could be due to bad detection, in this case fix errors by modifying "numbers" list)

# Variables
path = "book2.jpg"
height, width = 450, 450
model = initialize_prediction_model("digit_detection_model.h5")

print("loading model..")

# 0
img = cv2.imread(path)
img_blank = np.zeros((height, width, 3), np.uint8)

# 1
# getting grey, blur, and threshold imgs
processed = list(pre_processing(img))

# 2
# finding contours
img_contours = img.copy()
img_big_contours = img.copy()
img_c_only = img_blank.copy()
# getting contours from the threshold image since it is easier to find them in threshold
# RETR_EXTERNAL means outer contours
contours, hierarchy = cv2.findContours(processed[2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
cv2.drawContours(img_c_only, contours, -1, (0, 255, 0), 3)

# 3
# finding biggest contours
biggest, max_area = biggest_contour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(img_big_contours, biggest, -1, (0, 255, 0), 10)

    # aligning for transform
    pts_1 = np.float32(biggest)
    pts_2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # warping image instructions
    matrix = cv2.getPerspectiveTransform(pts_1, pts_2)

    # activating warp
    img_warp_colored = cv2.warpPerspective(img, matrix, (width, height))
    img_detected_digits = img_blank.copy()
    img_warp_gray = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)

    # 4
    # subdividing image
    img_solved_digits = img_blank.copy()
    boxes = split_boxes(img_warp_gray, 9)
    numbers = get_prediction(boxes, model)
    img_detected_digits = display_numbers(img_detected_digits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    pos_array = np.where(numbers > 0, 0, 1)

    # 5 solve
    board = np.array_split(numbers, 9)
    try:
        solve(board)
    except:
        print("error when solving")

    # flatten answer array
    flat = []
    for sublist in board:
        for item in sublist:
            flat.append(item)

    img_solved_digits = display_numbers(img_solved_digits, flat)


# creating arrays for stack_images()
img_array = (processed, [img_c_only, img_contours, img_warp_gray], [img_detected_digits, img_solved_digits, img_blank])
img_labels = (["x", "x", "x"], ["x", "x", "x"], ["x", "x", "x"])

# creating an image map using stack_images()
imgs = stack_images(img_array, 0.5)

# displaying image map
cv2.imshow("view", imgs)

cv2.waitKey(0)
