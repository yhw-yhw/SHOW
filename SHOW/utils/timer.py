"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import numpy as np
import torch
from mmcv import Timer
import contextlib
import time
from loguru import logger
import time

def tic():
    global start_time
    start_time = time.time()
    return start_time

def toc():
    if 'start_time' in globals():
        end_time = time.time()
        return end_time - start_time
    else:
        return None

@contextlib.contextmanager
def test_time():
    st=time.time()
    yield True
    et=time.time()
    logger.info(f'used time: {et-st}')
    


def timeit(func):
    def _warp(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elastic_time = time.time() - start_time
        print("The execution time of the function '%s'  is %.6fs\n" % (
            func.__name__, elastic_time))
        return result
    return _warp

class IterTimer:
    def __init__(self, name='time', sync=True, enabled=True,print_block=True):
        self.ll=20
        self.name = name
        self.times = []
        self.timer = Timer(start=False)
        self.sync = sync
        self.enabled = enabled
        self.print_block=print_block

    def __enter__(self):
        if self.print_block:
            title='START: '+self.name
            title=title.upper()
            logger.info('%'*(2*self.ll+len(title)))
            logger.info('%'*self.ll+title+'%'*self.ll)
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        self.timer.start()
        return self

    def __exit__(self, type, value, traceback):
        if self.print_block:
            title='EXIT: '+self.name
            title=title.upper()
            logger.info('%'*self.ll+title+'%'*self.ll)
            logger.info('%'*(2*self.ll+len(title)))
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        self.timer_record()
        self.timer._is_running = False
        self.print_time()

    def timer_start(self):
        self.timer.start()

    def timer_record(self):
        self.times.append(self.timer.since_last_check())

    def print_time(self):
        if not self.enabled:
            return
        logger.info('Average {} = {:.4f} seconds'.format(self.name, np.average(self.times)))
        logger.info('\n')

class IterTimers(dict):
    def __init__(self, *args, **kwargs):
        super(IterTimers, self).__init__(*args, **kwargs)
        # self.register_list=[]

    def disable_all(self):
        for timer in self.values():
            timer.enabled = False

    def enable_all(self):
        for timer in self.values():
            timer.enabled = True

    def add_timer(self, name='time', sync=True, enabled=True):
        self.__dict__[name] = IterTimer(
            name, sync=sync, enabled=enabled)
    
    def __getitem__(self, name):
        if not self.__dict__.get(name):
            name=name.upper()
            # logger.warning(f'timer "{name}" added')
            self.add_timer(name, sync=True, enabled=True)
        return self.__dict__[name]
    


default_timers = IterTimers()
