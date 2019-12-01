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
train_percent = int(sys.argv[4]) # percent of total data as training
val_percent = int(sys.argv[5]) # percent of leftover images as validation

sourcedir = os.path.abspath(os.path.join(script_dir , inputdir))
modedimagedir = os.path.abspath(os.path.join(script_dir , modedimagedir)) # gray, edge
outputdir = os.path.abspath(os.path.join(script_dir , outputdir))

print("Directories:")
print(sourcedir)
print(modedimagedir)
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
    # I first look underneath the first layer of the moded images directory
    moddedclasses = get_immediate_subdirectories(modedimagedir)
    # print("Modded class stuff")
    # print(moddedclasses)
    for amoded in moddedclasses:
        # Now we look at the classes underneath here - this could be gray, edge, or color being appended
        class_in_moded = os.path.abspath(os.path.join(modedimagedir , amoded))
        # print("Which class?")
        # print(class_in_moded)
        sub_classes = get_immediate_subdirectories(class_in_moded)
        for oneclass in sub_classes:
            if oneclass==imageclass:
                # This means we will not have duplicates
                # print("Which sub_class??")
                # print(oneclass)
                imagefolder_path = os.path.abspath(os.path.join(class_in_moded , oneclass))
                # Now we iterate over the list of files in this file location
                onlyfiles = [f for f in os.listdir(imagefolder_path) if os.path.isfile(os.path.join(imagefolder_path, f))]
                # print("All FILES!!!!!")
                # print(onlyfiles)
                # Now we iterate over all images in the directory
                for the_read_image in onlyfiles:
                    # create the image path
                    the_read_image_path = os.path.abspath(os.path.join(imagefolder_path , the_read_image ))
                    # print("Image full path")
                    # print(the_read_image_path)
                    # Now that we have gotten here, we have an image class type. This tells us where to put the image in test/val/train
                    # we now need to call a random number to decide if this object is a train, val, or test
                    result = random.random()
                    destpath=None
                    # Now we check which one we got
                    if 0 <= result < train_ratio:
                        # This means that we have an image that will go into the training directory
                        # We create the destination directory filepath
                        destpath = os.path.abspath(os.path.join(trainpath , oneclass ))
                        # now save the file to here.
                        #theimage = cv2.imread()
                    elif train_ratio <= result < train_ratio+val_ratio:
                        # This means we have to do image validation
                        destpath = os.path.abspath(os.path.join(valpath , oneclass ))
                    #if train_ratio+val_ratio <= result:
                    else:
                        # This means we are putting the file into the test directory
                        destpath = os.path.abspath(os.path.join(testpath , oneclass ))
                    # We have to append the filename from before to the filepath we are saving to
                    destpath = os.path.abspath(os.path.join(destpath , the_read_image ))
                    # print("Destination path:")
                    # print(destpath)
                    # Now that we have the source and destination, we can read in the file and save the file
                    input_image = cv2.imread(the_read_image_path)
                    status = cv2.imwrite(destpath, input_image)
                    # print("Save status"+str(status))
                    # exit()
print("Copying done.")