import imutils
from imutils.video import VideoStream
from pyzbar import pyzbar
import argparse
import datetime
import time
import cv2 as cv

# construct parser and parse args
parser = argparse.ArgumentParser()
default_path = 'barcodes.csv'
parser.add_argument('-o','--output',type=str,default=default_path,help='path for csv outfile')
args = vars(parser.parse_args())

# initialize video stream and allow the camera to warm up
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

# open the outfile
csv = open(args['output'],'w+')
found = set()

# run the stream
while True :
    # downsize to 400 px
    frame = vs.read()
    frame = imutils.resize(frame,width=400)

    barcodes = pyzbar.decode(frame)

    # loop over our barcodes
    for barcode in barcodes :
        # extract bounding box and draw it around the image
        x,y,w,h = barcode.rect
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

        # barcode objects have a bytes object that we need to
        # convert to a string
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type

        # draw the barcode data and type on the image
        text = f'{barcode_data},{barcode_type}\n'
        cv.putText(frame,text,(x,y-10),
                   cv.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),2)

        # if the barcode isnt already in our csv, write the
        # timestamp and barcode to the disk and put it in the set
        if barcode_data not in found :
            csv.write(f'{datetime.datetime.now()}, {barcode_data}')
            csv.flush()
            found.add(barcode_data)

    # show the output frame
    cv.imshow('Barcode Scanner',frame)
    key = cv.waitKey(1) & 0xFF

    # if the break key is pressed, exit loop
    break_key = 'q'
    if key == ord(break_key) : break

print('[INFO] cleaning up...')
csv.close()
cv.destroyAllWindows()
vs.stop()