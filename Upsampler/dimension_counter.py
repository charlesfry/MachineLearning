# here we will count images of each type of dimension and display the most common
import glob
from collections import Counter

from PIL import Image

DIR = './input/train2017/'

size_counts = Counter()

for filename in glob.glob(DIR+'*.jpg'): #assuming gif
    im=Image.open(filename)
    shape = im.size
    size_counts[shape] += 1


most_common = size_counts.most_common(10)

i = 0
for (k,v) in most_common :
    i += 1
    if i == 10 : break
    print(f'shape: {k}, count: {v}')

"""
results: 
shape: (640, 480), count: 25403
shape: (640, 427), count: 14596
shape: (480, 640), count: 8411
shape: (640, 426), count: 5353
shape: (500, 375), count: 5208
shape: (427, 640), count: 4259
shape: (640, 428), count: 3304
shape: (640, 425), count: 3134
shape: (612, 612), count: 2393
"""