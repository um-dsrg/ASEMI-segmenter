#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Johann A. Briffa <johann.briffa@um.edu.mt>

import sys
import os
import PIL.Image
import numpy as np
import argparse
import time
from datetime import timedelta

# image read/write functions

def getchannels(im):
   if ( im.mode.startswith('1') or
      im.mode.startswith('L') or im.mode.startswith('P') or
      im.mode.startswith('I') or im.mode.startswith('F') ):
      return 1
   if ( im.mode.startswith('RGBA') or im.mode.startswith('CMYK') ):
      return 4
   if ( im.mode.startswith('RGB') or im.mode.startswith('YCbCr') or
      im.mode.startswith('LAB') or im.mode.startswith('HSV') ):
      return 3
   # this should never happen
   return None

def imwrite(I,outfile):
   # Updated for PIL 1.1.6 upwards
   assert list(map(int, PIL.Image.VERSION.split('.'))) >= [1,1,6]
   # convert to image of the correct type based on shape and dtype
   im = PIL.Image.fromarray(I.squeeze())
   # save to file
   im.save(outfile, compression="tiff_lzw", optimize=True)
   return

def imread(infile):
   im = PIL.Image.open(infile)
   ch = getchannels(im)
   (x,y) = im.size
   # Updated for PIL 1.1.6 upwards
   # return array is read-only!
   assert list(map(int, PIL.Image.VERSION.split('.'))) >= [1,1,6]
   I = np.asarray(im).reshape(y,x,ch)
   return I

# method combining a set of individual-label images to a multi-ROI image

def combine_labels(input_files, output_file, soft):
   # read input images
   stack = [imread(infile).squeeze() for infile in input_files]
   # check for correct size and format
   assert all(len(im.shape) == 2 for im in stack) # single channel
   assert all(im.shape == stack[0].shape for im in stack) # same size
   # insert null label
   stack = [np.zeros(stack[0].shape, dtype=np.uint8)] + stack
   # rearrange format and check internal consistency
   stack = np.array(stack)
   if soft:
      stack = stack / 255.
      assert (stack.sum(axis=0)<=1).all() # probabilities sum to one or less
   else:
      assert np.logical_or(stack==0, stack==255).all() # binary masks
      stack = stack // 255
      assert (stack.sum(axis=0)<=1).all() # unique masks (not necessarily complete)
   # convert to multi-ROI representation
   I = stack.argmax(axis=0)
   # write output image
   imwrite(I.astype(np.uint8), output_file)
   return

## main program

def main():
   # interpret user options
   parser = argparse.ArgumentParser()
   parser.add_argument("-i", "--input", nargs='+', required=True,
                     help="input folders with single-label image stacks")
   parser.add_argument("-o", "--output", required=True,
                     help="output folder for multi-ROI image stack")
   parser.add_argument("-s", "--soft", action="store_true", default=False,
                     help="input images contain probabilities, not binary masks")
   args = parser.parse_args()

   # get lists of files in each folder
   filestack = [sorted(os.listdir(path)) for path in args.input]
   N = len(filestack[0])
   if not all(len(x)==N for x in filestack):
      print("Warning: not all folders contain an equal number of files")
   # iterate through each slice
   start = time.time()
   for i, images in enumerate(zip(*filestack)):
      print("Processing %d of %d..." % (i+1, N), end='')
      paths = [os.path.join(a,b) for a,b in zip(args.input, images)]
      combine_labels(paths, os.path.join(args.output, "%05d.tiff" % i), args.soft)
      print("ETA %s" % str(timedelta(seconds=(time.time()-start)*(N-i-1)/(i+1))))
   print("Time taken: %s" % str(timedelta(seconds=time.time()-start)))
   return

# main entry point
if __name__ == '__main__':
   main()
