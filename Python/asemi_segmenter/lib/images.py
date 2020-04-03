'''Image functions.'''

import subprocess
import tempfile
import platform
import PIL.Image
import numpy as np

#########################################
IMAGE_EXTS = set('tiff tif png jp2'.split(' '))


#########################################
def load_image(image_dir):
    '''
    Load an image file as an array. Converts the array to 16-bit first.

    :param str image_dir: The full file name (with path) to the image file.
    :return: The image array as 16-bit.
    :rtype: numpy.ndarray
    '''
    img_data = np.array(PIL.Image.open(image_dir))
    if img_data.shape == ():
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
                img_data = np.array(PIL.Image.open(os.path.join(tmp_dir, 'tmp.tif')))
        else:
            raise ValueError('Image format not supported.')
    
    #Convert to 16-bit.
    if img_data.dtype == np.uint32:
        img_data = np.right_shift(img_data, 8).astype(np.uint16)
    elif img_data.dtype == np.uint16:
        pass
    elif img_data.dtype == np.uint8:
        img_data = np.left_shift(img_data.astype(np.uint16), 8)

    return img_data


#########################################
def save_image(image_dir, image_data, compress=False):
    '''
    Save an image array to a file.

    :param str image_dir: The full file name (with path) to the new image file.
    :param numpy.ndarray image_data: The image array.
    '''
    options = dict()
    if image_dir.endswith('.tif') or image_dir.endswith('.tiff'):
        options['compression'] = 'tiff_deflate' if compress else None
    elif image_dir.endswith('.png'):
        options['compress_level'] = 9 if compress else 0
    
    image_format = 'I;16'
    if image_data.dtype == np.uint8:
        image_format = 'L'
    im = PIL.Image.fromarray(image_data, image_format)
    im.save(image_dir, **options)
