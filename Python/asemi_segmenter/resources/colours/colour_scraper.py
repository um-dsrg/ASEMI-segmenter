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

import urllib.request
import re
import json

with urllib.request.urlopen(
    'http://godsnotwheregodsnot.blogspot.com/2013/11/kmeans-color-quantization-seeding.html'
    ) as f:
    html = f.read().decode('utf-8')

colours_section = re.search(r'new String\[\]\{([^}]*)\}', html).group(1)
colour_strings = re.findall('#[0-9A-F]{6}', colours_section)
colour_rgbs = [
    [int(colour_string[2*i+1:2*i+3], 16) for i in range(3)]
    for colour_string in colour_strings
    ]

greys = list()
non_greys = list()
for colour_rgb in colour_rgbs:
    if max(
        abs(colour_rgb[0] - colour_rgb[1]),
        abs(colour_rgb[0] - colour_rgb[2]),
        abs(colour_rgb[1] - colour_rgb[2])
        ) <= 1:
        greys.append(colour_rgb)
    else:
        non_greys.append(colour_rgb)

output = (non_greys + greys)[:253]
with open('colours.json', 'w', encoding='utf-8') as f:
    json.dump(output, f)
