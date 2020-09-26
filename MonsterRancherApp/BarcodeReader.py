from pyzbar import pyzbar
import numpy as np
import cv2 as cv

def decode(img) :
    """
    find barcodes and QR codes
    :param img: the input image
    :return: decoded barcodes/QR codes
    """
    decoded_objects = pyzbar.decode(img)

    for obj in decoded_objects :
        print(f'Type: {obj.type}')
        print(f'Data: {obj.data}\n')

    return decoded_objects

def display(img,decoded_objects) :
    """

    :param img: .png image of a barcode
    :param decoded_objects: pyzbar.decode object
    :return:
    """
    for obj in decoded_objects :
        points = obj.polygon

        # if the points dont form a box, find convex hull
        if len(points) > 4 :
            hull = cv.convexHull(np.array([point for point in points],
                                          dtype=np.float32))
            hull = list(map(tuple,np.squeeze(hull)))

        else : hull = points

        # number of points in hull
        n = len(hull)

        # draw the convex hull
        for i in range(n) :
            cv.line(img,hull[i],hull[(i+1) % n], (255,0,0), 3)

    # display results
    cv.namedWindow('img_window',cv.WINDOW_NORMAL)
    resized_window = cv.resize(img,(400,400))
    cv.imshow('Results', resized_window)
    cv.waitKey(0)

if __name__ == '__main__' :
    img = cv.imread('./input/butt.jpg')
    decoded_objects = decode(img)
    display(img, decoded_objects)
