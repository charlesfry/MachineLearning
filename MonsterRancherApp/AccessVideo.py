import imutils
from imutils.video import VideoStream
from pyzbar import pyzbar
import argparse
import datetime
from time import time
import cv2 as cv

# construct parser and parse args
parser = argparse.ArgumentParser()
parser.add_argument('--o','output',type=str,default='barcodes.csv',help='path for csv outfile')
args = vars(parser.parse_args())

