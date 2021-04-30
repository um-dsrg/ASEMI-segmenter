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

'''Validation related functions.'''

import os
import pkg_resources
import json
import jsonref
import jsonschema
from asemi_segmenter.lib import files
from asemi_segmenter.lib import volumes

#########################################
def validate_json_with_schema_file(loaded_json, schema_name):
    '''
    Validate a loaded JSON object using a schema file in the json_schema resource directory.

    :param str loaded_json: The JSON object.
    :param str schema_name: The file name of the json_schema resource.
    :raise jsonschema.exceptions.ValidationError: Invalid JSON.
    '''
    schema = jsonref.loads(
        pkg_resources.resource_string('asemi_segmenter.resources.json_schema', schema_name).decode(),
        base_uri='file:///{}/'.format(
            pkg_resources.resource_filename('asemi_segmenter.resources.json_schema', '')
            ),
        jsonschema=True
        )
    jsonschema.Draft7Validator.check_schema(schema)
    validator = jsonschema.Draft7Validator(schema)
    error_message = None
    try:
        validator.validate(loaded_json)
    except jsonschema.exceptions.ValidationError as ex:
        error_message = ex.message
    if error_message is not None:
        raise ValueError(error_message)


#########################################
def validate_annotation_data(full_volume, subvolume_data, labels_data):
    '''
    Validate the preprocessed data of a full volume, with the meta data of a subvolume and labels.

    Validation consists of checking:
    * that the subvolume and all label slices are of the same shape as those of the full volume,
    * that the number of labels does not exceed the built in limit of FIRST_CONTROL_LABEL-1, and
    * that the number of slices in each label is equal to the number of slices in the subvolume.

    :param volumes.FullVolume full_volume: The full volume object from which to take the reference
        slice shape. If None then this will instead be taken from subvolume_data.
    :param volumes.VolumeData subvolume_data: The subvolume object to validate.
    :param list labels_data: A list of volumes.LabelData objects of different labels to validate.
    '''
    if full_volume is None:
        shape = subvolume_data.shape
    else:
        shape = full_volume.get_shape()[1:]

    if subvolume_data.shape != shape:
        raise ValueError('Subvolume slice shapes do not match volume slice shapes ' \
            '(volume={}, subvolume={}).'.format(
                shape, subvolume_data.shape
                ))

    if len(labels_data) > volumes.FIRST_CONTROL_LABEL:
        raise ValueError('Labels directory has too many labels ({}). ' \
            'Must be less than or equal to {}.'.format(
                len(labels_data), volumes.FIRST_CONTROL_LABEL-1
                ))

    for label_data in labels_data:
        if label_data.shape != shape:
            raise ValueError('Label {} slice shapes do not match volume slice shapes ' \
                '(volume={}, label={}).'.format(
                    label_data.name,
                    shape,
                    label_data.shape
                    ))
        if len(label_data.fullfnames) != len(subvolume_data.fullfnames):
            raise ValueError('Number of label slices ({}) in label {} does not equal number ' \
                'of slices in subvolume ({}).'.format(
                    len(label_data.fullfnames),
                    label_data.name,
                    len(subvolume_data.fullfnames)
                    ))


#########################################
def check_filename(fullfname, extension='', must_exist=False):
    '''
    Check that the file name is valid.

    Validation consists of checking:
    * that the directory to the file exists,
    * that the file name ends with extension, and
    * that the file exists if must_exist is true.

    :param str fullfname: The full file name (with path) to the file.
    :param bool must_exist: Whether the file should already exist or not.
    '''
    dir_path = os.path.split(fullfname)[0]
    if dir_path != '' and not files.fexists(dir_path):
        raise ValueError('File\'s directory does not exist.')
    if not fullfname.endswith(extension):
        raise ValueError('File\'s file name does not end with {}.'.format(extension))
    if must_exist and not files.fexists(fullfname):
        raise ValueError('File does not exist.')


#########################################
def check_directory(dir_path):
    '''
    Check that a directory is valid.

    Validation consists of checking:
    * that the directory exists.

    :param str dir_path: The path to the directory.
    '''
    if not files.fexists(dir_path):
        raise ValueError('Directory does not exist.')
