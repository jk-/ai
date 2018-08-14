#!/usr/bin/env python3

##################
# 2D Convolution #
##################

import matplotlib
matplotlib.use('Agg')

from skimage import io, viewer, color
import numpy as np

# blank picture
picture = io.imread('image.jpg')
picture = color.rgb2gray(picture)

# Because we need to account for the edges of the image we need
# to pad the picture with zeros
padded_picture = np.zeros((picture.shape[0] + 2, picture.shape[1] + 2))

# inject the picture inside the padded_picture
# python <3
padded_picture[1:-1,1:-1] = picture

# create an output the size of the picture
output = np.zeros_like(picture) 

# this kernel is used to sharpen the image
kernel = [
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
]

kernel = np.flipud(np.fliplr(kernel))

# this moves the kernel over each section of the picture
# and gets the summized dot product
for x in range(picture.shape[1]):
    for y in range(picture.shape[0]):
        output[y,x]=(kernel*padded_picture[y:y+3,x:x+3]).sum()

viewer.ImageViewer(output).show()
