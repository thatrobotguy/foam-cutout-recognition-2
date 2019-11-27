#!/usr/bin/env python3

# This script will take in 3 arguments: the input empty_set image, the foam image, and then the output image.
import os, cv2, copy, math, random, sys, numpy, glob, itertools

# Where is this scritp currently running
script_dir = os.path.dirname(os.path.abspath(__file__))

# print("Arguments")
# print(str(len(sys.argv)))
# print(str(sys.argv))

"""
This scripts takes 2 arguments as input:
The first one is the input directory. This directory contains all of the images that need to be resized.
The second argument is the directory that will contain all of the images that were resized.
The third and fourth argument are the height and width (h,w)
"""

if not len(sys.argv) == 5:
    print("Not enough input arguments.")
    print("Usage: python3 thescript.py local/source/fir local/output/dir height width")
    exit()

inputdir = sys.argv[1]
outputdir= sys.argv[2]
image_h = int(sys.argv[3]) # width
image_w = int(sys.argv[4]) # height

# inputdir = script_dir +"/" + inputdir
# outputdir= script_dir +"/" + outputdir

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

# Here is a function to get the immediate subdirectories from the root source directory
# https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_immediate_subdirectories(a_dir):
    return sorted([name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))])

# Given some extensions in a list, get all files in the directors
# https://stackoverflow.com/questions/48001890/how-to-read-images-from-a-directory-with-python-and-opencv
def getFilenames(exts):
    fnames = [glob.glob(ext) for ext in exts]
    fnames = list(itertools.chain.from_iterable(fnames))
    return fnames

# Now that we know we have the direcotry to the source data, we need to traverse the source directory for all the classes inside of it.
the_sub_dirs = get_immediate_subdirectories(sourcedir)
print("Sub-directories of the root source folder.")
print(the_sub_dirs)
# We now need to join the input source path dir with the class directories to create the output directories
for aclass in the_sub_dirs:
    # These are creating the gray and color directories
    part_gray_path  = os.path.abspath(os.path.join(outputdir , "gray"))
    part_color_path = os.path.abspath(os.path.join(outputdir , "color"))
    full_out_gray_path = os.path.abspath(os.path.join( part_gray_path , aclass))
    full_out_color_path = os.path.abspath(os.path.join( part_color_path , aclass))
    if not os.path.isdir(full_out_gray_path):
        # Make the directory
        print("Creating direcotry: " + str(full_out_gray_path))
        os.makedirs(full_out_gray_path)
    if not os.path.isdir(full_out_color_path):
        # Make the directory
        print("Creating direcotry: " + str(full_out_color_path))
        os.makedirs(full_out_color_path)

print("Now creating modded images.")
gray = "gray"
color = "color"
counter = 0
# Now that the directories are made, we need to create the resized color and resized gray images.
# iterate over the image classes
for theclass in the_sub_dirs:
    # create the source image path
    image_path = os.path.abspath(os.path.join(sourcedir , theclass))
    # print("Source Image path")
    # print(image_path)
    # list all image filenames in the object class
    thefiles = sorted(os.listdir(image_path))
    # print("The directories")
    # print(thefiles)
    # Now we iterate through all of the files in this class directory
    for image in thefiles:
        if image.endswith(".JPG") or image.endswith(".jpg"):
            # counter = counter + 1
            # print("COUNTER: "+str(counter))
            # read the image
            # print("Image")
            # print(image)
            # print(image_path)
            full_image_path = os.path.join(image_path, image)
            # print("FULL source image path")
            # print(full_image_path)
            input_img = cv2.imread(full_image_path)
            # print("Image shape")
            # print(input_img.shape)
            #
            # create the resized image
            input_img_resize = cv2.resize(input_img, (image_h, image_w))
            # convert the resized image to gray
            input_img_resize_gray = cv2.cvtColor(input_img_resize, cv2.COLOR_BGR2GRAY)
            # Now we generate the filepaths that the modded images will be saved to
            out_part_gray_path  = os.path.abspath(os.path.join(outputdir , gray))
            out_part_color_path = os.path.abspath(os.path.join(outputdir , color))
            out_full_out_gray_path = os.path.abspath(os.path.join(  out_part_gray_path , theclass))
            out_full_out_color_path = os.path.abspath(os.path.join( out_part_color_path , theclass))
            # Now we append the image name to the filepath
            out_full_out_gray_path = str(os.path.abspath(os.path.join(  out_full_out_gray_path  , image)))
            out_full_out_color_path = str(os.path.abspath(os.path.join( out_full_out_color_path , image)))
            # print("New Image paths")
            # print(out_full_out_gray_path)
            # print(out_full_out_color_path)
            status1 = cv2.imwrite(out_full_out_color_path , input_img_resize)
            status2 = cv2.imwrite(out_full_out_gray_path  , input_img_resize_gray)
            # print("Files saved: "+str(status1)+" _ "+str(status2))
            # cv2.imshow("blah", input_img_resize_gray)
            # cv2.waitKey(0)

print("Program complete.")