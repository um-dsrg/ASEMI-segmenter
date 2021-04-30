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

'''Image functions.'''

import subprocess
import tempfile
import platform
import io
import PIL.Image
import numpy as np

#########################################
IMAGE_EXTS_IN = set('.tiff .tif .png .jp2'.split(' '))
IMAGE_EXTS_OUT = set('.tiff .tif .png'.split(' '))


#########################################
def matplotlib_to_imagedata(figure):
    '''
    Convert a Matplotlib figure to a numpy array.

    :param matplotlib.pyplot.Figure figure: The figure to convert.
    :return: The numpy array.
    :rtype: numpy.ndarray
    '''
    with io.BytesIO() as io_buf:
        figure.savefig(io_buf, format='raw')
        io_buf.seek(0)
        image_data = np.frombuffer(
            io_buf.getvalue(),
            dtype=np.uint8
            ).reshape(
                int(figure.bbox.bounds[3]),
                int(figure.bbox.bounds[2]),
                4
                )[:,:,:3]
    return image_data


#########################################
def check_image_ext(fname, image_exts):
    '''
    Check if the file name extension of an image is one of the extensions in image_exts.

    :param str fname: The file name of the image.
    :param set image_exts: A set of file extensions with the dot included e.g. {'.png'}.
        Alternatively just use images.IMAGE_EXTS_IN or images.IMAGE_EXTS_OUT.
    '''
    return fname[-4:] in image_exts or fname[-5:] in image_exts


#########################################
def convert_dtype(image_data, num_bits=16):
    '''
    Convert image data type.

    :param numpy.ndarray image_data: The image array.
    :param int num_bits: The number of bits per pixel to force the image data into.
        Must be 8, 16, or 32
    '''
    if num_bits not in {8, 16, 32}:
        raise ValueError('num_bits must be 8, 16, or 32.')

    if num_bits == 8:
        if image_data.dtype == np.uint8:
            pass
        elif image_data.dtype == np.uint16:
            image_data = np.right_shift(image_data, 16-8).astype(np.uint8)
        elif image_data.dtype == np.uint32:
            image_data = np.right_shift(image_data, 32-8).astype(np.uint8)
        else:
            raise NotImplementedError('Image datatype not supported.')
    elif num_bits == 16:
        if image_data.dtype == np.uint8:
            image_data = np.left_shift(image_data.astype(np.uint16), 16-8)
        elif image_data.dtype == np.uint16:
            pass
        elif image_data.dtype == np.uint32:
            image_data = np.right_shift(image_data, 32-16).astype(np.uint16)
        else:
            raise NotImplementedError('Image datatype not supported.')
    elif num_bits == 32:
        if image_data.dtype == np.uint8:
            image_data = np.left_shift(image_data.astype(np.uint32), 32-8)
        elif image_data.dtype == np.uint16:
            image_data = np.left_shift(image_data.astype(np.uint32), 32-16)
        elif image_data.dtype == np.uint32:
            pass
        else:
            raise NotImplementedError('Image datatype not supported.')
    else:
        raise NotImplementedError('Number of bits not supported.')

    return image_data


#########################################
def load_image(image_dir, num_bits=16):
    '''
    Load an image file as an array. Converts the array to 16-bit first.

    :param str image_dir: The full file name (with path) to the image file.
    :param int num_bits: The number of bits per pixel to force the image data into.
        Must be 8, 16, or 32
    :return: The image array as 16-bit.
    :rtype: numpy.ndarray
    '''
    if num_bits not in {8, 16, 32}:
        raise ValueError('num_bits must be 8, 16, or 32.')
    if not check_image_ext(image_dir, IMAGE_EXTS_IN):
        raise ValueError('Image is not of an accepted extension.')

    image_data = np.array(PIL.Image.open(image_dir))
    if image_data.shape == ():
        if image_dir.endswith('.jp2') and platform.system() == 'Linux':
            with tempfile.TemporaryDirectory(dir='/tmp/') as tmp_dir: #Does not work on Windows!
                subprocess.check_output(
                    [
                        'opj_decompress',
                        '-i', image_dir,
                        '-o', os.path.join(tmp_dir, 'tmp.tiff')  #Uncompressed image output for speed.
                        ],
                    stderr=subprocess.DEVNULL
                    )
                image_data = np.array(PIL.Image.open(os.path.join(tmp_dir, 'tmp.tif')))
        else:
            raise ValueError('Image format not supported.')

    image_data = convert_dtype(image_data, num_bits)

    return image_data


#########################################
def save_image(image_dir, image_data, num_bits=16, compress=False):
    '''
    Save an image array to a file.

    :param str image_dir: The full file name (with path) to the new image file.
    :param int num_bits: The number of bits per pixel to force the image data into.
        Must be 8, 16, or 32
    :param numpy.ndarray image_data: The image array.
    '''
    if num_bits not in {8, 16, 32}:
        raise ValueError('num_bits must be 8, 16, or 32.')
    if not check_image_ext(image_dir, IMAGE_EXTS_OUT):
        raise ValueError('Image is not of an accepted extension.')

    options = dict()
    if image_dir.endswith('.tif') or image_dir.endswith('.tiff'):
        options['compression'] = 'tiff_deflate' if compress else None
    elif image_dir.endswith('.png'):
        options['compress_level'] = 9 if compress else 0

    image_data = convert_dtype(image_data, num_bits)
    im = PIL.Image.fromarray(image_data)

    im.save(image_dir, **options)
