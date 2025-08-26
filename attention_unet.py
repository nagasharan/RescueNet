from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D,\
                             Dropout, Layer, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow.image as tfimg

import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import math
import random


filename = "./testset/test-org-img/10801.jpg"
img = Image.open(filename)
width,height  = img.size
print(width,height)
img.show()