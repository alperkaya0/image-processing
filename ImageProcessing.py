import math

import cv2
import numpy as np


# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") no need to use this anymore

# Right now, basically my program takes anime scene and saves manga version of it at RESULT folder
def texture_segmenting(img_name, standard, binary, sig, adaptive_thresh_size):
    # img = cv2.imread("images/person.jpg")
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
    square_size = 3
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
            standard_deviation = standard
            if not standard_deviation:
                if not len(pixel_values) == 0:
                    value = int(max(pixel_values) - min(pixel_values))
            elif standard_deviation:
                if not len(pixel_values) == 0:
                    u = 0
                    total = 0
                    for i in pixel_values:
                        u = u + i
                    u = u / len(pixel_values)
                    for i in pixel_values:
                        total = total + (u - i) ** 2
                        # total is actually result of standard deviation
                    try:
                        if (not (len(pixel_values) - 1 <= 0)):
                            total = math.sqrt(total / len(pixel_values) - 1)
                        elif total >= 0 and not len(pixel_values) == 1:
                            total = math.sqrt(total)
                    except:
                        pass
                    pixel_values.clear()
                    value = int(total)
            last_values.append(value)
            pixel_values.clear()
            cv2.circle(img, (y, x), 1, (value, value, value), -1)

    #for list_number, list_ in enumerate(masked_values):
    #    for coordinate in list_:
    #        if list_number > mask_number/2 or list_number-1 == mask_number:
    #            value = (list_number + 1) * mask_number
    #            cv2.circle(img, coordinate, 1, (value, value, value), -1)

    for x in range(x_shape):
        for y in range(y_shape):
            img[x, y] = 255 - img[x, y]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    c_equ = clahe.apply(l)
    img = cv2.merge((c_equ, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    binarize = binary
    sigmoid = sig
    mean_std_values_list = []
    mean_std_values = 0
    adaptive_thresholding_square_size = adaptive_thresh_size
    for i, y in enumerate(last_values):
        if y % adaptive_thresholding_square_size == 0:
            for multiplier in range(adaptive_thresholding_square_size):
                for K in range(adaptive_thresholding_square_size):
                    mean_std_values = mean_std_values + last_values[y + K + multiplier * (y_shape - 1)]
            mean_std_values_list.append(mean_std_values / (adaptive_thresholding_square_size ** 2))
            mean_std_values = 0
    mean_std_values_list.reverse()
    for val in range(len(last_values)):
        if binarize:
            if val == 0 and len(mean_std_values_list) > 0:
                mean_std_values = mean_std_values_list.pop()
            elif (val % 9 == 0) and len(mean_std_values_list) > 0:
                mean_std_values = mean_std_values_list.pop()
            if not sigmoid:
                if last_values[val] > mean_std_values:
                    print("Work in progress... " + str(last_values[val]))
                    last_values[val] = 255
                elif last_values[val] < mean_std_values:
                    print("Work in progress... " + str(last_values[val]))
                    last_values[val] = 0
            else:
                last_values[val] = last_values[val] - mean_std_values
                last_values[val] = 1 / (1 + math.exp(-last_values[val]))
                if last_values[val] > 0.5:
                    print("Work in progress... " + str(last_values[val]))
                    last_values[val] = 255
                else:
                    print("Work in progress... " + str(last_values[val]))
                    last_values[val] = 0
    print("lastvalueslist", len(last_values))
    if binarize:
        last_values.reverse()
        for x in range(x_shape):
            for y in range(y_shape):
                value = last_values.pop()
                # print("work in progress..." + str(value))
                cv2.circle(img, (y, x), 1, (value, value, value), -1)
    print("lastvalueslist", len(last_values))
    return img, standard_deviation, binarize, sigmoid


def save_file(standard, binr, sigmoid, img, name, extension, adaptive_threshold_size):
    cv2.imwrite("Results/Standard deviation " + str(standard) + " Binarized " + str(binr) + " Sigmoid " + str(
        sigmoid) + " " + "Adaptive Thresholding Square Size " + str(
        adaptive_threshold_size) + " " + name + "." + extension, img)


def save_all(name, extension, adaptive_size):
    img, standard, binr, sigmoid = texture_segmenting(name + "." + extension, False, False, False, adaptive_size)
    save_file(standard, binr, sigmoid, img, name, extension, adaptive_size)
    
    img, standard, binr, sigmoid = texture_segmenting(name + "." + extension, True, False, False, adaptive_size)
    save_file(standard, binr, sigmoid, img, name, extension, adaptive_size)

    img, standard, binr, sigmoid = texture_segmenting(name + "." + extension, True, True, False, adaptive_size)
    save_file(standard, binr, sigmoid, img, name, extension, adaptive_size)

    img, standard, binr, sigmoid = texture_segmenting(name + "." + extension, True, True, True, adaptive_size)
    save_file(standard, binr, sigmoid, img, name, extension, adaptive_size)

    img, standard, binr, sigmoid = texture_segmenting(name + "." + extension, False, True, True, adaptive_size)
    save_file(standard, binr, sigmoid, img, name, extension, adaptive_size)

    img, standard, binr, sigmoid = texture_segmenting(name + "." + extension, False, True, False, adaptive_size)
    save_file(standard, binr, sigmoid, img, name, extension, adaptive_size)

name = input("Enter name : ")
ext = input("Enter extension without point : ")
save_all(name, ext, 15)
