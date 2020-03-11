import datetime
import timeit

#########################################
def get_timestamp():
    return str(datetime.datetime.now())

#########################################
def get_readable_duration(duration_seconds):
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
    
    #########################################
    def __init__(self):
        self._start = 0
        self.duration = 0
        self.runs = 0
    
    #########################################
    def __enter__(self):
        self._start = timeit.default_timer()
        self.runs += 1
        return self
    
    #########################################
    def __exit__(self, type, value, traceback):
        self.duration += timeit.default_timer() - self._start
    
    #########################################
    def reset(self):
        self.duration = 0
        self.runs = 0
        
    #########################################
    def get_current_duration(self):
        return timeit.default_timer() - self._start
