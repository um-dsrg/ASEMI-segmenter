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
    def __init__(self, label_names, skip_colours=0):
        '''
        Constructor.

        :param list label_names: A list of label names in a desired order such that
            the index of the label name is the label index of that label.
        :param int skip_colours: The number of colours in the sequence to skip
            (colours are in a fixed sequence).
        '''
        self.label_names = label_names

        self.names_palette = {
            label: tuple(cmap[i + skip_colours])
            for (i, label) in enumerate(label_names)
            }
        if len(self.names_palette) != len(self.label_names):
            raise ValueError('Cannot have duplicate label names.')
        self.index2colour = np.array([
            self.names_palette[label]
            for label in self.label_names
            ], np.uint8)
        self.colour2index = {
            self.names_palette[label]: i + skip_colours
            for (i, label) in enumerate(self.label_names)
            }

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
        :return: Numpy array of same shape as label_indexes plus an extra dimension for
            the RGB channels.
        :rtype: numpy.ndarray
        '''
        return self.index2colour[label_indexes]

    #########################################
    def colours_to_label_indexes(self, colours):
        '''
        Convert an 8-bit RGB array to an array of label indexes.

        :param numpy.ndarray colours: Numpy array (of any shape) of RGB channels.
        :return: Numpy array of same shape as colours minues the extra dimension
            for the RGB channels.
        :rtype: numpy.ndarray
        '''
        result = np.empty(colours.shape[:-1], np.uint8)
        for (i, _) in np.ndenumerate(result):
            result[i] = self.colour2index[tuple(colours[i].tolist())]
        return result
