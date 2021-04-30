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

'''Timings related module.'''

import datetime
import timeit

#########################################
def get_timestamp():
    '''
    Get a string timestamp of the current date and time.

    :return: The timestamp.
    :rtype: str
    '''
    return str(datetime.datetime.now())

#########################################
def get_readable_duration(duration_seconds):
    '''
    Convert a number of seconds into a readable string duration broken into days, hours, etc.

    :param int duration_seconds: Number of seconds to convert.
    :return: The readable duration.
    :rtype: str
    '''
    d = round(duration_seconds)
    days = d//(24*60*60)
    hours = d%(24*60*60)//(60*60)
    minutes = d%(24*60*60)%(60*60)//60
    seconds = d%(24*60*60)%(60*60)%60
    return '{}{}{}{}'.format(
            ('{:d}d '.format(days) if days > 0 else ''),
            ('{:0>2d}h:'.format(hours) if days > 0 or hours > 0 else ''),
            ('{:0>2d}m:'.format(minutes) if days > 0 or hours > 0 or minutes > 0 else ''),
            '{:0>2d}s'.format(seconds),
        )

#########################################
class Timer(object):
    '''
    A stopwatch type object for measuring durations.

    This object is meant to be used in a 'with' context, for example:

    .. code-block:: python
        with Timer() as t:
            #do something
        print(t.duration)  # Duration in seconds.

    It is also possible to use the same Timer object multiple times to get the total duration of
    all times, for example:

    .. code-block:: python
        t = Timer()
        with t:
            #do something
        with t:
            #do something else
        print(t.runs)  # 2 runs.
        print(t.duration)  # Total duration of both runs in seconds.
    '''

    #########################################
    def __init__(self):
        '''Constructor.'''
        self._start = 0
        self.duration = 0
        self.runs = 0

    #########################################
    def __enter__(self):
        '''Start the stopwatch.'''
        self._start = timeit.default_timer()
        self.runs += 1
        return self

    #########################################
    def __exit__(self, type, value, traceback):
        '''Pause the stopwatch.'''
        self.duration += timeit.default_timer() - self._start

    #########################################
    def reset(self):
        '''Reset the stopwatch.'''
        self.duration = 0
        self.runs = 0

    #########################################
    def get_current_duration(self):
        '''
        Get the current duration in seconds, mid run (before the end of the with block).

        :return: The current duration.
        :rtype: int
        '''
        return timeit.default_timer() - self._start
