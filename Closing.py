import math

import cv2
import numpy as np



img_name = input("File full name : ")


img = cv2.imread(img_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
c_equ = clahe.apply(l)
gray = cv2.merge((c_equ, a, b))
gray = cv2.cvtColor(gray, cv2.COLOR_LAB2BGR)
cv2.imwrite("Results/Clahe_Applied" + " " + img_name, gray)

gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
cv2.imwrite("Results/Gray" + " " + img_name, gray)

x_shape, y_shape = len(gray), len(gray[0])
pixel_values = []
square_size = int(input("Enter square size: "))
value = 0
last_values = []
standard_deviation = False
mask_number = 12
mask_val = int(255 / mask_number)
masked_values = []
for i in range(mask_number):
    masked_values.append([])
for x in range(x_shape):
    for y in range(y_shape):
        gray[x, y] = 255 - gray[x, y]
for x in range(x_shape):
    for y in range(y_shape):
        for it in range(mask_number):
            if it * mask_val < gray[x, y] < (it + 1) * mask_val:
                masked_values[it].append((y, x))

        # cv2.circle(img, (x, y), 1, (255, 0, 255), -1)
        for x_ in range(square_size):
            for y_ in range(square_size):
                if not (x_ == int(square_size / 2 + 1) and y_ == int(square_size / 2 + 1)):
                    addition_x = x_
                    addition_y = y_
                    if x_ < int(square_size / 2) + 1: addition_x = -x_
                    if y_ < int(square_size / 2) + 1: addition_y = -y_
                    if (
                            x + addition_x < x_shape - 1 and x + addition_x > 0 and y + addition_y < y_shape - 1 and y + addition_y > 0):
                        pixel_values.append(gray[x + addition_x, y + addition_y])
        if not len(pixel_values) == 0:
            value = int(min(pixel_values))
        last_values.append(value)
        pixel_values.clear()
        cv2.circle(img, (y, x), 1, (value, value, value), -1)
cv2.imwrite("Results/CLOSED " + img_name, img)