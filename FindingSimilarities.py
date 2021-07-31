import math
import random
import cv2
import numpy as np

img_name = input("File full name : ")
img_name2 = input("Second file's ful name : ")
gray_bool = input("Gray ? Y/N : ")
if gray_bool == "Y":
    gray_bool = True
else:
    gray_bool = False

img = cv2.imread(img_name)
img2 = cv2.imread(img_name2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

x_shape, y_shape = len(gray), len(gray[0])
x_shape2, y_shape2 = len(gray2), len(gray2[0])

desired_x = min([x_shape2, x_shape])
desired_y = min([y_shape2, y_shape])

if not gray_bool:
    img = cv2.resize(img, [desired_y, desired_x])
    img2 = cv2.resize(img2, [desired_y, desired_x])

    img_channels = cv2.split(img)
    img_channels2 = cv2.split(img2)

    value = 0
    i = 0

    for i in [0, 1, 2]:
        for x in range(desired_x):
            for y in range(desired_y):
                if (img_channels[i][x, y] == img_channels2[i][x, y]):
                    value = img_channels[i][x, y]
                else:
                    value = 0
                print(value)
                img_channels[i][x, y] = value
                img_channels2[i][x, y] = value

    img = cv2.merge(img_channels)
    img2 = cv2.merge(img_channels2)

    cv2.imwrite("Results/INTERSECTIONS " + img_name, img)
    cv2.imwrite("Results/INTERSECTIONS " + img_name2, img2)
elif gray_bool:
    gray = cv2.resize(gray, [desired_y, desired_x])
    gray = gray[0:(desired_x), 0:(desired_y)]
    gray2 = cv2.resize(gray2, [desired_y, desired_x])
    gray2 = gray2[0:(desired_x), 0:(desired_y)]

    value = 0
    i = 0

    for x in range(desired_x):
        for y in range(desired_y):
            if gray[x, y] == gray2[x, y]:
                value = gray[x,y]
            else:
                value = 0


            print(value)
            gray[x, y] = value
            gray2[x, y] = value

    cv2.imwrite("Results/INTERSECTIONS GRAY " + img_name, gray)
    cv2.imwrite("Results/INTERSECTIONS GRAY " + img_name2, gray2)
input("Press ENTER to exit...")
