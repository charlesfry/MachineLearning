import numpy as np
import pandas as pd
from PIL import Image
import os
import sys
from pathlib import Path


try:
    os.makedirs('./input/train_scale/')
except OSError:
    pass

test_path = './input/test_scale/'

keep_size = (640, 480)

kept_files = 0
for _,_,files in os.walk(test_path) :
    for x in files :
        img = Image.open(x)
        if img.size != keep_size : continue
        print(img.filename)
        kept_files += 1

print(kept_files)


def main() :
    pass

if __name__ == '__main__' :
    main()