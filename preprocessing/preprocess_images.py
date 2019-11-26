# CS549 - Computer Vision Final Project - Deep Learning Approach
# Griffin Bishop, Nick St. George,  Luke Ludington, Andrew Schueler

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import random

# Path to data folder. Data folder shoulder have a "resize" folder
# "Reize" folder should have 1 folder for each of the 6 classes.
path = "./data/"

percentage = 0.25
batch_size = 1

size = 500

transform = transforms.Compose([
    transforms.Resize(size),  #maintains aspect ratio
#    transforms.CenterCrop(size),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(os.path.join(path, "resize/"), transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)


# img.permute(1, 2, 0)
# Permute rearranges image dimensions (e.g. [3, w, h] -> [w, h, 3])

os.mkdir(os.path.join(path, "out/"))
os.mkdir(os.path.join(path, "out/train/"))
os.mkdir(os.path.join(path, "out/val/"))

i = 0
for inputs, labels in dataloader:
    for img, label in zip(inputs, labels):
        filename = str(i)+".jpg"
        directory = "train" if random.random() > percentage else "val"
        directory = "out/"+directory+"/"+str(label.numpy())+"/"
        dirpath = os.path.join(path, directory)

        _, h, w = img.shape
        m = max(h, w)

        img = transforms.ToPILImage()(img)
        img_ = transforms.Pad((0, m-h, m-w, 0))(img)
        img_ = transforms.Resize(size)(img_)
        img_ = transforms.ToTensor()(img_)

        #plt.imshow(img_.permute(1, 2, 0))
        #plt.show()
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)

        torchvision.utils.save_image(img_, os.path.join(dirpath, filename))
        i += 1
