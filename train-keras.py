import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import os


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        fill_mode='nearest')
