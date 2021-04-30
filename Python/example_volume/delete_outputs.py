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

import sys
import os
import shutil
from asemi_segmenter.lib import files

print('Deleting...')
print(' output/checkpoint.json')
files.delete('output/checkpoint.json')
print(' output/log.txt')
files.delete('output/log.txt')
for dir in os.listdir('output'):
    dir = 'output/{}'.format(dir)
    if os.path.isdir(dir):
        print('', dir)
        for fname in os.listdir(dir):
            if fname != '.gitignore':
                files.delete('{}/{}'.format(dir, fname))
print('Ready.')
