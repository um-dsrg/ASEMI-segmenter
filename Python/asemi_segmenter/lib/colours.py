'''Colour related functions.'''

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches

#########################################
cmap = plt.get_cmap('nipy_spectral')

#########################################
class LabelPalette(object):
    '''Create a colour palette in 8-bit RGB, one colour for each label.'''
    
    #########################################
    def __init__(self, label_names):
        '''
        Constructor.
    
        :param list label_names: A list of label names in a desired order such that
            the index of the label name is the label index of that label.
        '''
        self.label_names = label_names
        
        self.names_palette_float = {
            label: cmap((i + 1)/(1 + len(label_names) + 1))[:3]
            for (i, label) in enumerate(label_names)
            }
        self.names_palette_int = {
            label: tuple(
                round(channel*255)
                for channel in self.names_palette_float[label]
                )
            for (i, label) in enumerate(label_names)
            }
        
        self.index_palette_float = [
            self.names_palette_float[label]
            for label in self.label_names
            ]
        self.index_palette_int = [
            self.names_palette_int[label]
            for label in self.label_names
            ]
    
    #########################################
    def get_legend(self):
        '''
        Get a matplotlib figure showing a legend of colours to label names.
        
        :return: Matplotlib figure with just a legend.
        :rtype: matplotlib.pyplot.Figure
        '''
        (fig, ax) = plt.subplots(1, 1)
        ax.legend(
            handles=[
                matplotlib.patches.Patch(color=self.names_palette_float[label], label=label)
                for label in self.label_names
                ],
            loc='center'
            )
        ax.axis('off')
        return fig

    #########################################
    def label_indexes_to_colours(self, label_indexes):
        '''
        Convert an array of label indexes to an 8-bit RGB array.
        
        :param numpy.ndarray label_indexes: Numpy array (of any shape) of integer label
            indexes.
        :return: Numpy array of same shape plus an extra dimension for the RGB channels.
        :rtype: numpy.ndarray
        '''
        return np.array(self.index_palette_int, np.uint8)[label_indexes]