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
parser.add_argument('--o','output',type=str,default=default_path,help='path for csv outfile')
args = vars(parser.parse_args())

# initialize video stream and allow the camera to warm up
print('[INFO] starting video stream...')
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# open the outfile
csv = open(args['output'],'w+')
found = set()