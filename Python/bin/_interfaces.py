import tqdm
import textwrap
import os
import sys
from asemi_segmenter.lib import times
from asemi_segmenter import listener

#########################################
TEXT_WIDTH = 100

#########################################
class CliProgressListener(listener.ProgressListener):
    
    #########################################
    def __init__(self, log_file_fullfname):
        self.prog = None
        self.prog_prev_value = 0
        self.log_file_fullfname = log_file_fullfname
    
    #########################################
    def print_(self, text):
        print(text)
        if self.log_file_fullfname is not None:
            with open(self.log_file_fullfname, 'a', encoding='utf-8') as f:
                print(text, file=f)
    
    #########################################
    def log_output(self, text):
        if text == '':
            self.print_('')
        else:
            for (i, line) in enumerate(textwrap.wrap(text, TEXT_WIDTH)):
                if i == 0:
                    self.print_(line)
                else:
                    self.print_('   '+line)
    
    #########################################
    def error_output(self, text):
        self.print_('ERROR: ' + text)
        self.print_(times.get_timestamp())
    
    #########################################
    def current_progress_start(self, start, total):
        self.prog = tqdm.tqdm(initial=start, total=total)
        self.prog_prev_value = start
    
    #########################################
    def current_progress_update(self, curr):
        self.prog.update(curr - self.prog_prev_value)
        self.prog_prev_value = curr
    
    #########################################
    def current_progress_end(self):
        self.prog.close()
