'''Module for progress observer model functions.'''

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