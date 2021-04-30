#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
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

import os
import math
import PIL.Image
import numpy as np
import argparse
from asemi_segmenter.lib import images
from asemi_segmenter.lib import files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
        help='Input folder with multi-ROI image stack.')
    parser.add_argument('--output', required=True,
        help='Output folder for single-label image stacks.')
    parser.add_argument('--labels_fullfname', required=True,
        help='Full file name (with path) to text file containing the label names in the order of the ROI.')
    parser.add_argument('--bits', required=True, type=int, choices=[8, 16],
        help='The number of bits to save the output images as.')
    parser.add_argument('--image_extension', required=True, choices=['png', 'tiff', 'tif'],
        help='The filename extension of the output images.')
    args = parser.parse_args()

    print('Running...')

    if not files.fexists(args.input):
        raise ValueError('Input directory does not exist.')

    fullfnames = []
    with os.scandir(args.input) as it:
        for entry in it:
            if entry.name.startswith('.'):
                continue
            if not images.check_image_ext(entry.name, images.IMAGE_EXTS_IN):
                continue
            if entry.is_file():
                fullfnames.append(os.path.join(args.input, entry.name))
    if not fullfnames:
        raise ValueError('Input directory does not have any images.')
    fullfnames.sort()

    slice_shape = None
    for fullfname in fullfnames:
        with PIL.Image.open(fullfname) as f:
            shape = (f.height, f.width)
            if f.mode[0] not in 'LI':
                raise ValueError('Found slice that is not a greyscale image ' \
                    '({}).'.format(fullfname))
        if slice_shape is not None:
            if shape != slice_shape:
                raise ValueError('Found differently shaped slices ' \
                    '({} and {}).'.format(
                        fullfnames[0], fullfname
                        ))
        else:
            slice_shape = shape

    num_digits_in_filename = math.ceil(math.log10(len(fullfnames)+1))

    with open(args.labels_fullfname, 'r', encoding='utf-8') as f:
        labels = f.read().strip().split('\n')
    for label in labels:
        files.mkdir(os.path.join(args.output, label))

    for (i, fullfname) in enumerate(fullfnames):
        print('Separating {}'.format(fullfname))
        with PIL.Image.open(fullfname) as f:
            f.load()
            img = np.array(f)
        # The index 0 in the image is the null label which is ignored
        assert np.max(img) <= len(labels), (np.max(img), len(labels))
        for (index, label) in zip(range(1, len(labels)+1), labels):
            mask = img == index
            if args.bits == 8:
                mask = mask*(2**8 - 1)
                mask = mask.astype(np.uint8)
            elif args.bits == 16:
                mask = mask*(2**16 - 1)
                mask = mask.astype(np.uint16)

            images.save_image(
                os.path.join(
                    os.path.join(args.output, label),
                    '{}_{:0>{}}.{}'.format(
                        label,
                        i + 1,
                        num_digits_in_filename,
                        args.image_extension
                        )
                    ),
                mask,
                num_bits=args.bits,
                compress=True
                )

if __name__ == '__main__':
   main()
