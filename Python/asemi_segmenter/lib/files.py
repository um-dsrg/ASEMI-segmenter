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

'''Module for file related functions.'''

import os, errno
import shutil

#########################################
def mkdir(dir):
    '''
    Create a directory recursively and ignore if it already exists.

    :param str dir: The directory path to create.
    '''
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

#########################################
def delete(path):
    '''
    Delete a file or directory and ignore if it already exists.

    :param str path: The path to the object to delete.
    '''
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError as e:
        pass

#########################################
def fexists(path):
    '''
    Check if a file or directory exists.

    :param str path: The path to the object to check.
    '''
    return os.path.isdir(path) or os.path.isfile(path)
