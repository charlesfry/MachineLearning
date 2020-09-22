import numpy as np
import cv2 as cv
import imutils

def detect(img) :
    # convert to greyscale for ease of computation
    gscale = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # compute Scharr gradient magnitude representation
    ddepth = cv.CV_32F if imutils.is_cv2() else cv.CV_32F
    grad_X = cv.Sobel(gscale,ddepth=ddepth,dx=1,dy=0,ksize=-1)
    grad_Y = cv.Sobel(gscale,ddepth=ddepth,dx=0,dy=1,ksize=-1)

    # subtract y-grad from X-grad
    grad = cv.subtract(grad_X,grad_Y)
    grad = cv.convertScaleAbs(grad)

    # blur and threshold the image
    blurred = cv.blur(grad,(9,9))
    _,thresh = cv.threshold(blurred,255,255,cv.THRESH_BINARY)

    # construct a closing kernel to apply to the thresholded image
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(21,7))
    closed = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel)

    # perform erosions and dialations
    closed = cv.erode(closed,None,iterations=4)
    closed = cv.dilate(closed,None,iterations=4)

    # find the contours in the thresholded image
    conts = cv.findContours(closed.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)

    # if no contours found
    if len(conts) < 1 : return None

    # if we got this far, we found at least 1 contour
    # get bounding box of largest contour
    largest_cont = sorted(conts,key=cv.contourArea,reverse=True)[0]
    rect = cv.minAreaRect(largest_cont)
    box = cv.boxPoints(rect) if imutils.is_cv2() else cv.boxPoints(rect)
    box = np.int0(box)

    # return barcode bounding box
    return box