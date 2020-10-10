from PIL import Image
import numpy as np
import pandas as pd
import os
from pathlib import Path

DIR_IN = Path('./input/train2017/')
REDUCED_SIZE = Path('./input/scale_train')

batch_size = 32

