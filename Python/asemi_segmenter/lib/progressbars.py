'''
Module for displaying the progress of a process.
'''

import os
import timeit
from asemi_segmenter.lib import times

#########################################
class ProgressBar(object):
    '''Class for keeping track of progress and displaying information about it.'''

    #########################################
    def __init__(self, initial_iter, final_iter, max_iter_times=-1, print_output=True, log_listener=lambda i, duration:None):
        '''
        Constructor.

        :param int initial_iter: The initial iteration number.
        :param int final_iter: The final iteration number.
        :param int max_iter_times: The number of iteration durations from the end to
            use to measure the average speed.
        :param bool print_output: Whether to actually use Python prints.
        :param callable listener: A listener for logging the progress of the progress bar.
            Signature of the listener is (i, duration) -> None where i is the current
            iteration number and duration is the duration of said iteration.
        '''
        self.initial_iter = initial_iter
        self.final_iter = final_iter
        self.max_iter_times = max_iter_times

        self.curr_iter = None
        self.final_iter_num_digits = len(str(self.final_iter))
        self.start_time = None
        self.prev_time = None
        if max_iter_times == -1:
            self.iter_times = None
        else:
            self.iter_times = list()
        self.total_iter_times = 0.0
        self.iter_times_count = 0
        self.print_output = print_output
        self.log_listener = log_listener

    #########################################
    def __show_bar(self):
        '''
        Show the progress bar.
        '''
        perc = self.curr_iter/self.final_iter

        if self.iter_times_count > 0:
            avg_iter_duration = self.total_iter_times/self.iter_times_count
            time_left = (self.final_iter - self.curr_iter)*avg_iter_duration

            time_elapsed_str = times.get_readable_duration(timeit.default_timer() - self.start_time)
            time_left_str = times.get_readable_duration(time_left)
            if avg_iter_duration > 999.99:
                avg_iter_duration_str = ' '*6
            else:
                avg_iter_duration_str = '{: >6.2f}'.format(avg_iter_duration)
        else:
            time_elapsed_str = times.get_readable_duration(0)
            time_left_str = '??s'
            avg_iter_duration_str = ' '*6

        pre_bar_line = '{: >4.0%}|'.format(perc)

        post_bar_line = '|{: >{}d}/{: >{}d} {} ETA:{}@{}s/it'.format(
            self.curr_iter, self.final_iter_num_digits,
            self.final_iter, self.final_iter_num_digits,
            time_elapsed_str,
            time_left_str,
            avg_iter_duration_str
            )

        bar_len = os.get_terminal_size().columns - len(pre_bar_line) - len(post_bar_line)
        if bar_len < 0:
            post_bar_line = '|'
            bar_len = os.get_terminal_size().columns - len(pre_bar_line) - len(post_bar_line)

        if bar_len < 0:
            line = '{: >4.0%}'.format(perc)
        else:
            amount_bar_filled = round(perc*bar_len)
            amount_bar_empty = bar_len - amount_bar_filled
            line = '{}{}{}{}'.format(
                pre_bar_line,
                '█'*amount_bar_filled,
                ' '*amount_bar_empty,
                post_bar_line
                )

        if self.print_output:
            print('\r'+line, end='')

    #########################################
    def init(self):
        '''
        Show the initial progress bar.
        '''
        self.prev_time = self.start_time = timeit.default_timer()
        self.curr_iter = self.initial_iter
        if self.max_iter_times != -1:
            self.iter_times = list()
        self.total_iter_times = 0.0
        self.iter_times_count = 0
        self.__show_bar()

    #########################################
    def update(self, iterations=1):
        '''
        Update the progress bar with a new iteration.
        '''
        if self.prev_time is None:
            raise Exception('Cannot call update() before calling init().')

        time_now = timeit.default_timer()
        iter_duration = (time_now - self.prev_time)/iterations
        for _ in range(iterations):
            self.curr_iter += 1
            if self.max_iter_times == -1:
                self.total_iter_times += iter_duration
                self.iter_times_count += 1
            else:
                if len(self.iter_times) == self.max_iter_times:
                    self.iter_times.pop(0)
                self.iter_times.append(iter_duration)
            self.log_listener(self.curr_iter, iter_duration)
        if self.max_iter_times != -1:
            self.total_iter_times = sum(self.iter_times)
            self.iter_times_count = len(self.iter_times)
        self.prev_time = time_now

        self.__show_bar()

    #########################################
    def close(self):
        '''
        Close the progress bar.
        '''
        if self.print_output:
            print()
