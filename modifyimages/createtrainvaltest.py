#!/usr/bin/env python3

# This script will take in 3 arguments: the input empty_set image, the foam image, and then the output image.
import os, cv2, copy, math, random, sys, numpy, glob, itertools

# Where is this scritp currently running
script_dir = os.path.dirname(os.path.abspath(__file__))

