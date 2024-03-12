import time
from functools import wraps
import random

import torch
import numpy as np


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        t1 = time.time()
        print('Delta-t for {}: {} s'.format(func.__name__, t1 - t0))
        return ret
    return wrapper


def class_timeit(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        ret = func(self, *args, **kwargs)
        t1 = time.time()
        print('Delta-t for {}: {} s'.format(func.__name__, t1 - t0))
        return ret
    return wrapper


def set_all_random_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_gpu_memory(show=False):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    if show:
        print('Device 0 memory info:')
        print('    {} MB total'.format(t / 1024 ** 2))
        print('    {} MB reserved'.format(r / 1024 ** 2))
        print('    {} MB allocated'.format(a / 1024 ** 2))
    return t, r, a

