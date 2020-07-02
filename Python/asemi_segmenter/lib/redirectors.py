import sys

#########################################
class PrintRedirector(object):
    '''Redirect all prints within a 'with' block to a listener.'''

    #########################################
    def __init__(self, listener):
        '''
        Constructor.

        :param callable listener: The listener to receive the redirection.
            Listener should accept one string argument.
        :param str prefix: A prefix to put in front of each output.
        '''
        class RedirectorStream(object):
            def write(self, text):
                if listener is None or text == '\n':
                    return
                tmp = sys.stdout
                sys.stdout = sys.__stdout__
                listener(text)
                sys.stdout = tmp
            def flush(self):
                pass

        self.stream = RedirectorStream()


    #########################################
    def __enter__(self):
        '''Redirect prints to listener.'''
        sys.stdout = self.stream

    #########################################
    def __exit__(self, etype, ex, traceback):
        '''Restore ordinary print functionality.'''
        sys.stdout = sys.__stdout__
