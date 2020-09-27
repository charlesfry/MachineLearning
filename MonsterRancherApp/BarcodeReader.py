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
    cv.imshow('Results',img)
    cv.waitKey(0)

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

if __name__ == '__main__' :
    img = cv.imread('./input/personbar.jpg')
    img = ResizeWithAspectRatio(img,width=400)
    decoded_objects = decode(img)
    display(img, decoded_objects)
