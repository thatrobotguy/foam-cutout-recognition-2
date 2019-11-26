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

# import ipdb; ipdb.set_trace()

# Top level data directory.
# This folder must include "train" and "val" directories,
# each with a folder for each class.
data_dir = "./preprocessing/data/out/"

num_classes = 6
batch_size = 24
num_epochs = 25

show_shape = False

train_xform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])
val_xform = transforms.Compose([
    transforms.ToTensor()
])


train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_xform)
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"),   val_xform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, num_workers=4, shuffle=True)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def init_model():

    layers = []
                # Conv2d is (input channels, output channels, kernel size)
    layers.append(nn.Conv2d(3, 32, 3))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2)) # (kernel size, stride, padding,...)

    layers.append(nn.Conv2d(32, 32, 3))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2)) # (kernel size, stride, padding,...)

    layers.append(nn.Conv2d(32, 64, 3))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2)) # (kernel size, stride, padding,...)

    layers.append(nn.Conv2d(64, 32, 3))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2)) # (kernel size, stride, padding,...)

    # Shape of activation after this last layer:
    layers.append(Flatten()) # I let pytorch compute 26912 for us
    layers.append(nn.Linear(26912, 4096))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(p=0.25))
    layers.append(nn.Linear(4096, 512))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(p=0.25))
    layers.append(nn.Linear(512, num_classes))
    #layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)



model = init_model()

print(model)

model = model.to(device)

observations = [305, 188, 250, 112, 73, 71]
class_weights = [sum(observations)/x for x in observations]
class_weights = torch.FloatTensor(class_weights).cuda()

criterion = nn.CrossEntropyLoss(class_weights)
optimizer = optim.Adam(model.parameters())

model.train()



modulo = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        if show_shape:
            print(outputs.shape)
            exit()

        _, predictions = torch.max(outputs, 1) # max function returns both values and indices
        running_corrects += torch.sum(predictions == labels.data)

        # Add to confusion matrix
        for i, prediction in enumerate(predictions):
            confusion_matrix[prediction,labels.data[i]] += 1

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_dataloader.dataset)
    epoch_acc = running_corrects.double() / len(train_dataloader.dataset)
    print(f"Epoch {epoch} loss: {epoch_loss:.6f}  |  (train set) accuracy: {epoch_acc:.4f}")
    print(confusion_matrix)

def validate():
    confusion_matrix = np.zeros((num_classes, num_classes))
    running_corrects = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1) # max function returns both values and indices
            running_corrects += torch.sum(predictions == labels.data)

            # Add to confusion matrix
            for i, prediction in enumerate(predictions):
                confusion_matrix[prediction,labels.data[i]] += 1

    accuracy = running_corrects.double() / len(val_dataloader.dataset)
    return confusion_matrix, accuracy

model.eval()
confusion, accuracy = validate()
print(f"Validation accuracy: {accuracy:.4f}")
print("First dim: What we predicted. Second dim: what the class actually was")
print(confusion)
