'''Colour related functions.'''

import pkg_resources
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches

#########################################
cmap = json.loads(
    pkg_resources.resource_string(
        'asemi_segmenter.resources.colours', 'colours.json'
        ).decode()
    )

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
        
        self.names_palette = {
            label: tuple(cmap[i])
            for (i, label) in enumerate(label_names)
            }
        self.index2colour = np.array([
            self.names_palette[label]
            for label in self.label_names
            ], np.uint8)
    
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
                matplotlib.patches.Patch(
                    color=tuple(channel/255 for channel in self.names_palette[label]),
                    label=label
                    )
                for label in self.label_names
                ],
            loc='center',
            prop={'size': 16}
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
        return self.index2colour[label_indexes]