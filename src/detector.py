import cv2
import pyzbar.pyzbar as pyzbar
import os
import numpy as np
from pyzbar.pyzbar import ZBarSymbol

INPUT_PATH = '../images'
OUTPUT_PATH = '../QRCodes'


def save_qrcode(im, decodedObjects, file_name):
    for decodedObject in decodedObjects:
        points = decodedObject.polygon

        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        # Number of points in the convex hull
        print(hull)
        x_min = hull[0].x
        x_max = hull[0].x
        y_min = hull[0].y
        y_max = hull[0].y
        for tmp_point in hull[1:]:
            if tmp_point.x > x_max:
                x_max = tmp_point.x
            if tmp_point.x < x_min:
                x_min = tmp_point.x
            if tmp_point.y > y_max:
                y_max = tmp_point.y
            if tmp_point.y < y_min:
                y_min = tmp_point.y
        print(y_min, y_max, x_min, x_max)
        # todo: enningxie/bug
        cv2.imwrite(os.path.join(OUTPUT_PATH, file_name), im[y_min-5: y_max+5, x_min-5: x_max+5])


def detect(im, file_name):
    decoded_objects = pyzbar.decode(im, symbols=[ZBarSymbol.QRCODE])
    if len(decoded_objects):
        save_qrcode(im, decoded_objects, file_name)
        return True
    else:
        return False


def zoom_op(im, size):
    return cv2.resize(im, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)


if __name__ == '__main__':
    for file_name in os.listdir(INPUT_PATH):
        # read image
        inputImage = cv2.imread(os.path.join(INPUT_PATH, file_name))
        detected_flag = detect(inputImage, file_name)
        if detected_flag:
            continue
        for tmp_size in np.arange(1.1, 2.1, 0.1):
            zoomed_Image = zoom_op(inputImage, tmp_size)
            detected_flag = detect(zoomed_Image, file_name)
            if detected_flag:
                print('{}: {}.'.format(file_name, tmp_size))
                break
