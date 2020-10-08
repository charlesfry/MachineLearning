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