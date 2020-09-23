from pyzbar import pyzbar
import argparse
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('-i','--image',required=True,
                    help='path to input image')
args = vars(parser.parse_args())

# load image
img = cv.imread(args['image'])

barcodes = pyzbar.decode(image=img)

# loop over all barcodes
for code in barcodes :
    # get bounding box
    x,y,w,h = code.rect
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    # barcodes are bytes objects, so convert to string
    code_data = code.data.decode('utf-8')
    code_type = code.type

    # draw barcode data and type on the image
    text = f'{code_data}, {code_type}'
    cv.putText(img,text,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,
               .5,(0,0,255),2)

    # output barcode type and data to terminal
    print(f'[INFO] Found {code_type} barcode: {code_data}')

    # show image
    cv.imshow('Image',img)
    cv.waitKey(0)
