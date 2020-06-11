'''Module for progress observer model functions.'''
import textwrap
import tqdm

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
    def current_progress_start(self, start, total):
        '''
        Listener for initialising a progress bar for a sub-stage.

        :param int start: The starting iteration in the progress.
            Normally 0, but can be more if a checkpoint is resumed.
        :param int total: The total number of iterations in the sub-stage.
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
    def __init__(self, log_file_fullfname=None, text_width=100):
        self.prog = None
        self.prog_prev_value = 0
        self.log_file_fullfname = log_file_fullfname
        self.text_width = text_width

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
            for (i, line) in enumerate(textwrap.wrap(text, self.text_width)):
                if i == 0:
                    self.print_(line)
                else:
                    self.print_('    '+line)

    #########################################
    def error_output(self, text):
        self.print_('ERROR: ' + text)

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
