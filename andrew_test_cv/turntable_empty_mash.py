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

# Because My laptop does not have a 4k screen, we will shrink the empty set image
empty = cv2.resize(empty, (1736, 1024))

# This is the square size of the foam image
foam_size = 300
# I first need to resize the foam image - this is just for this script's experiment.
foam_resize = cv2.resize(foam, (foam_size,foam_size))
# Now I need to get the dimensions of the enpty set image.
h, w, _ = empty.shape
print(str(empty.shape))
# Now I need to figure out where to put the foam image in the empty set image
# I calculate the h and w offset for the foam image
off_h = int(random.random() * float( h-foam_size - 1))
off_w = int(random.random() * float( w-foam_size - 1))
# Now I need to create an image mask for the empty set image
empty_mask = numpy.zeros((h,w), dtype=numpy.uint8)
# Now we set the pixels in the mask that are 1's rather than 0s
empty_mask[off_h:off_h+foam_size, off_w:off_w+foam_size] = 255
# Now we invert the mask
empty_mask = cv2.bitwise_not(empty_mask)
# Now that we have the mask, lets apply it to the empty set image
empty_image = cv2.bitwise_and(empty, empty, mask=empty_mask)

# Now we create the resized foam image with the same dimensions as the empty set image
big_foam_img = numpy.zeros(empty.shape, dtype=numpy.uint8)
# Then we place the foam image onto the blown up foam image
big_foam_img[off_h:off_h+foam_size, off_w:off_w+foam_size, :] = foam_resize
# Now we need to insert the foam image into the empty image
result_image = big_foam_img + empty_image
# Show the image
# cv2.imshow('Original Image', result_image)
# cv2.waitKey(0)
# exit()

# Save the image
cv2.imwrite(resultname, result_image)
