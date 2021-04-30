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

'''Module for progress observer model functions.'''

import textwrap
from asemi_segmenter.lib import progressbars

#########################################
class ProgressListener(object):
    '''Class for listening to the progress of segmenter commands.'''

    #########################################
    def __init__(self):
        '''Empty constructor.'''
        pass

    #########################################
    def log_output(self, text):
        '''
        Listener for each line of text logging the command's activity.

        :param str text: The line of log text.
        '''
        pass

    #########################################
    def error_output(self, text):
        '''
        Listener for fatal errors.

        :param str text: The error message.
        '''
        pass

    #########################################
    def overall_progress_start(self, total):
        '''
        Listener for initialising a progress bar for the whole command's activity.

        :param int total: The total number of stages in the command.
        '''
        pass

    #########################################
    def overall_progress_update(self, curr, status):
        '''
        Listener for the start of a stage in the whole command's activity.

        :param int curr: The number of the stage that has started.
        :param str status: A short text description of the stage that has started.
        '''
        pass

    #########################################
    def overall_progress_end(self):
        '''
        Listener for the destruction of the progress bar for the whole command's activity.
        '''
        pass

    #########################################
    def current_progress_start(self, start, total, unstable_time=False):
        '''
        Listener for initialising a progress bar for a sub-stage.

        :param int start: The starting iteration in the progress.
            Normally 0, but can be more if a checkpoint is resumed.
        :param int total: The total number of iterations in the sub-stage.
        :param bool unstable_time: Whether the duration of each iteration is stable.
        '''
        pass

    #########################################
    def current_progress_update(self, curr):
        '''
        Listener for the completion of an iteration in the sub-stage.

        :param int curr: The number of the iteration that has completed.
        '''
        pass

    #########################################
    def current_progress_end(self):
        '''
        Listener for the destruction of the progress bar for the sub-stage.
        '''
        pass


#########################################
class CliProgressListener(ProgressListener):

    #########################################
    def __init__(self, log_file_fullfname=None, text_width=100, print_output=True):
        self.prog = None
        self.last_prog_curr = None
        self.log_file_fullfname = log_file_fullfname
        self.text_width = text_width
        self.print_output = print_output
        if self.log_file_fullfname is not None:
            def progressbar_listener(i, duration):
                with open(self.log_file_fullfname, 'a', encoding='utf-8') as f:
                    print('', i, duration, sep='\t', file=f)
            self.progressbar_listener = progressbar_listener
        else:
            self.progressbar_listener = lambda i, duration:None

    #########################################
    def print_(self, text):
        if self.print_output:
            print(text)
        if self.log_file_fullfname is not None:
            with open(self.log_file_fullfname, 'a', encoding='utf-8') as f:
                print(text, file=f)

    #########################################
    def log_output(self, text):
        if text == '':
            self.print_('')
        else:
            for (i, line) in enumerate(textwrap.wrap(text, self.text_width)):
                if i == 0:
                    self.print_(line)
                else:
                    self.print_('    '+line)

    #########################################
    def error_output(self, text):
        self.print_('ERROR: ' + text)

    #########################################
    def current_progress_start(self, start, total, unstable_time=False):
        self.prog = progressbars.ProgressBar(
            start,
            total,
            max_iter_times=5 if unstable_time else -1,
            print_output=self.print_output,
            log_listener=self.progressbar_listener
            )
        self.prog.init()
        if self.log_file_fullfname is not None:
            with open(self.log_file_fullfname, 'a', encoding='utf-8') as f:
                print('', 'iteration', 'duration (s)', sep='\t', file=f)
        self.last_prog_curr = start

    #########################################
    def current_progress_update(self, curr):
        self.prog.update(curr - self.last_prog_curr)
        self.last_prog_curr = curr

    #########################################
    def current_progress_end(self):
        self.prog.close()
        self.prog = None
        self.last_prog_curr = None
