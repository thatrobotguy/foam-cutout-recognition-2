#!/usr/bin/env python3

# This script will take in 3 arguments: the input empty_set image, the foam image, and then the output image.
import os, cv2, copy, math, random, sys, numpy

print("Arguments")
print(str(len(sys.argv)))
print(str(sys.argv))
print("Done.")

"""
This script needs to take in an empty set image and a foam image.
The script will then overlay the foam image over the empty set image.
I will set the foam image to 300x300 for testing purposes only.
The resized foam image will be overlayed onto the empty set image in a random location.
The resulting image will be the same size as the original empty set image.
"""

empty = cv2.imread(sys.argv[1])
foam  = cv2.imread(sys.argv[2])
resultname = sys.argv[3]

# This is the square size of the foam image
foam_size = 300
# I first need to resize the foam image - this is just for this script's experiment.
foam_resize = cv2.resize(foam, (foam_size,foam_size))
# Now I need to get the dimensions of the enpty set image.
h, w, _ = empty.shape
print(str(empty.shape))
print("Dims")
print(str(h))
print(str(w))
# Now I need to figure out where to put the foam image in the empty set image
# I calculate the h and w offset for the foam image
off_h = int(random.random() * float( h-foam_size - 1))
off_w = int(random.random() * float( w-foam_size - 1))
# Now I need to create an image mask for the empty set image
empty_mask = numpy.zeros((h,w), dtype=numpy.uint8)
# Now we set the pixels in the mask that are 1's rather than 0s
empty_mask[off_h:off_h+foam_size, off_w:off_w+foam_size] = 1
# Now that we have the mask, lets apply it to the empty set image
empty_image = cv2.bitwise_and(empty, empty, mask=empty_mask)
# Show the image
cv2.imshow('Original Image', empty)
cv2.waitKey(0)
exit()
