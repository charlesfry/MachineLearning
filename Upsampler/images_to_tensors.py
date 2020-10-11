import numpy as np
import tensorflow as tf
from PIL import Image
import os
from tqdm import tqdm



train_ds = []
train_labels = []

for root,dirs,files in os.walk('./input/scale_data/train/train_scale/') :
    progress = tqdm(total=len(files),position=0)
    for file in files :
        filename = file.split('/')[-1]
        path = os.path.join(root,file)
        img = Image.open(path)
        pix = tf.keras.preprocessing.image.img_to_array(img)
        test_file_path = f'./input/scale_data/test/test_scale/{filename}'
        test_path = test_file_path
        img_test = Image.open(test_path)
        pix_test = tf.keras.preprocessing.image.img_to_array(img_test)

        train_ds.append(pix)
        train_labels.append(pix_test)
        progress.update(1)



