#!/usr/bin/env python3

# This script will take in 3 arguments: the input empty_set image, the foam image, and then the output image.
import os, cv2, copy, math, random, sys, numpy, glob, itertools

# Where is this scritp currently running
script_dir = os.path.dirname(os.path.abspath(__file__))

if not len(sys.argv) == 6:
    print("Not enough input arguments.")
    print("Usage: python3 thescript.py local/source/fir local/output/dir trainpercent valpercent")
    exit()

inputdir = sys.argv[1] # where did the images originaly from
modedimagedir= sys.argv[2] # Where are the gray, color, and edge images
outputdir= sys.argv[3] # Where are the test, val, and training directories supposed to be located?
train_percent = int(sys.argv[3]) # percent of total data as training
val_percent = int(sys.argv[4]) # percent of leftover images as validation

sourcedir = os.path.abspath(os.path.join(script_dir , inputdir))
outputdir = os.path.abspath(os.path.join(script_dir , outputdir))

print("Directories:")
print(sourcedir)
print(outputdir)

# does the root source image directory exist?
if not os.path.exists(inputdir):
    # Then we don't have the images that we need.
    print("No Image data directory found.")
    exit()

# Does the root image output directory exist?
if not os.path.exists(outputdir):
    # Then we make the directory
    print("Output directory does not exist. Creating the directory.")
    os.mkdir(outputdir)

# Now we will create the test, val, and train directories
trainpath = os.path.abspath(os.path.join(outputdir , "train"))
valpath = os.path.abspath(os.path.join(outputdir , "val"))
testpath = os.path.abspath(os.path.join(outputdir , "test"))

# Does the root image output directory exist?
if not os.path.exists(trainpath):
    # Then we make the directory
    print("Train directory does not exist. Creating the directory.")
    os.mkdir(trainpath)

# Does the root image output directory exist?
if not os.path.exists(valpath):
    # Then we make the directory
    print("Val directory does not exist. Creating the directory.")
    os.mkdir(valpath)

# Does the root image output directory exist?
if not os.path.exists(testpath):
    # Then we make the directory
    print("Test directory does not exist. Creating the directory.")
    os.mkdir(testpath)

# Here is a function to get the immediate subdirectories from the root source directory
# https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_immediate_subdirectories(a_dir):
    return sorted([name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])

# Now that we know we have the direcotry to the source data, we need to traverse the source directory for all the classes inside of it.
the_sub_dirs = get_immediate_subdirectories(sourcedir)
print("Sub-directories of the root source folder.")
print(the_sub_dirs)

# Now we have to make the directories that go underneath train val test
for theclass in the_sub_dirs:
    # We check the directory existence
    trainclass = os.path.abspath(os.path.join(trainpath , theclass))
    valclass =   os.path.abspath(os.path.join(valpath , theclass))
    testclass =  os.path.abspath(os.path.join(testpath , theclass))
    if not os.path.exists(trainclass):
        print("Making directory:\t"+str(trainclass))
        os.mkdir(trainclass)
    if not os.path.exists(valclass):
        print("Making directory:\t"+str(valclass))
        os.mkdir(valclass)
    if not os.path.exists(testclass):
        print("Making directory:\t"+str(testclass))
        os.mkdir(testclass)

# Now that the directories for the classes have been created, we need to start filling them with images
# we are given the percentages of train, val, and test
train_ratio = float(train_percent) / 100.0
val_ratio   = (float(val_percent) / 100.0) * (1.0 - train_ratio )
test_ratio  = 1.0 - (train_ratio + val_ratio)

print("Training percent: "+str(train_ratio))
print("Validate percent: "+str(val_ratio))
print("Testing  percent: "+str(test_ratio))

# Now that we have the ratios, we can now actually start
for imageclass in the_sub_dirs:
    # We now need to load an image from the source class dir and randomly choose if it goes in 


# Now that we know the classes of the files, we will extract the images from the processed image classes and start assigning them to train val test
