import time, copy
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

"""
This is the class definition for the transform that takes in an image and then spits out the binary image that shows edges in the image.
"""

class DetectEdge(object):
    """Simply get the gray and then edge-detected image.

    Args:
        No arguments needed. Image size is implied when passing the image into the filters.
    """

    # def __init__(self):
    # def __init__(self, output_size):
        # assert isinstance(output_size, (int, tuple))
        # if isinstance(output_size, int):
        #     # self.output_size = (output_size, output_size)
        #     # set the default dimensions to be square with 3 channels
        #     self.output_size = (3, output_size, output_size)
        # else:
        #     # assert len(output_size) == 2
        #     assert len(output_size) == 3
        #     self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # h, w = image.shape[:2]
        # new_h, new_w = self.output_size

        # top = np.random.randint(0, h - new_h)
        # left = np.random.randint(0, w - new_w)

        # This is the grayscale conversion
        grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # This is the conversion from grayscale to an edge detected image
        edge_img = cv2.Canny(grayimage,181,210)

        # This is the crop image
        """image = image[top: top + new_h, 
                      left: left + new_w]"""

        # We don't want to modify the landmarks
        # landmarks = landmarks - [left, top]

        # return {'image': image, 'landmarks': landmarks}
        return {'image': edge_img, 'landmarks': landmarks}
