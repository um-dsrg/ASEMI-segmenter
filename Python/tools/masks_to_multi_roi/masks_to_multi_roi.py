#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Johann A. Briffa
#
# This file is part of ASEMI-segmenter.
#
# ASEMI-segmenter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ASEMI-segmenter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASEMI-segmenter.  If not, see <http://www.gnu.org/licenses/>.


import sys
import os
import math
import PIL.Image
import numpy as np
import argparse
import time
from datetime import timedelta
import multiprocessing as mp

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
        if not (stack.sum(axis=0)<=1.).all():
            print("Warning: probabilities sum to %f" % stack.sum(axis=0).max())
    else:
        assert np.logical_or(stack==0, stack==255).all() # binary masks
        stack = stack // 255
        assert (stack.sum(axis=0)<=1).all() # unique masks (not necessarily complete)
    # convert to multi-ROI representation
    I = stack.argmax(axis=0)
    # write output image
    imwrite(I.astype(np.uint8), output_file)
    return

# worker thread

def worker(queue_in, queue_out, soft, force, dry_run):
    # iterate through slices to process
    for i, input_files, output_file in iter(queue_in.get, None):
        # skip if output already exists
        if os.path.isfile(output_file) and not force:
            queue_out.put("skipped slice %i - output exists" % i)
            continue
        # skip if it's a dry run
        if dry_run:
            queue_out.put("would process slice %i" % i)
            continue
        # do the work
        combine_labels(input_files, output_file, soft)
        queue_out.put("processed slice %i" % i)
    # tell user we're done
    print("Process %d exiting" % mp.current_process().pid)
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
    parser.add_argument("-f", "--force", action="store_true", default=False,
        help="regenerate output even if it already exists")
    parser.add_argument("-n", "--dry-run", action="store_true", default=False,
        help="show what will happen but do not create output")
    parser.add_argument("-p", "--processes", type=int, default=1,
        help="number of parallel processes")
    args = parser.parse_args()

    print('Running...')

    # get lists of files in each folder
    labels = [path.strip(os.path.sep).split(os.path.sep)[-1] for path in args.input]
    filestack = [sorted(os.listdir(path)) for path in args.input]
    N = len(filestack[0])
    if not all(len(x)==N for x in filestack):
        print("Warning: not all folders contain an equal number of files")

    # save labels
    if not args.dry_run:
        with open(os.path.join(args.output, 'labels.txt'), 'w', encoding='utf-8') as f:
            for label in labels:
                print(label, file=f)

    # main timer
    start = time.time()

    # set up queues and spawn a pool of workers
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    for i in range(args.processes):
        mp.Process(target=worker, args=(queue_in, queue_out, args.soft, args.force, args.dry_run)).start()

    # iterate through each slice
    num_digits_in_filename = math.ceil(math.log10(N+1))
    for (i, images) in enumerate(zip(*filestack)):
        paths = [os.path.join(a,b) for a,b in zip(args.input, images)]
        output = os.path.join(
                    args.output,
                    "{:0>{}d}.tiff".format(i + 1, num_digits_in_filename)
                    )
        print("Scheduling %d of %d..." % (i+1, N))
        queue_in.put( (i, paths, output) )

    # wait on completion
    for i in range(N):
        result = queue_out.get()
        ETA = str(timedelta(seconds=(time.time()-start)*(N-i-1)/(i+1)))
        print("Completed %d of %d, ETA %s (%s)" % (i+1, N, ETA, result))

    # tell workers to stop
    for i in range(args.processes):
        queue_in.put(None)

    # all done
    print("Time taken: %s" % str(timedelta(seconds=time.time()-start)))
    return

# main entry point
if __name__ == '__main__':
   main()
