import math
import cv2
import numpy as np


def apply_clahe(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    c_equ = clahe.apply(l)
    gray = cv2.merge((c_equ, a, b))
    gray = cv2.cvtColor(gray, cv2.COLOR_LAB2BGR)
    cv2.imwrite("Results/Clahe_Applied"+ " " + img_name, gray)


apply_clahe("irmak.jpg")