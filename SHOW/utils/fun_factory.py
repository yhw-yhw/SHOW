import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from loguru import logger



def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


class func_factory(object):
    def __init__(self,factory_name='task'):
        self.factory_name=factory_name
        self.__func_name_to_mem_map={}
        
    def register_module(self):
        def decorator(func):
            logger.info(f'{func.__name__} is registered')
            self.__func_name_to_mem_map[func.__name__] = func
            def wrapper(*args, **kwargs):
                func(*args, **kwargs)
            return wrapper
        return decorator

    def build(self,fun_name):
        if fun_name not in self.__func_name_to_mem_map.keys():
            logger.warning(f'fun_name {fun_name} not registered, return a empty func')
            def empty_func(*args, **kwargs):
                pass
            return empty_func
        return self.__func_name_to_mem_map[fun_name]

default_func_factory=func_factory('default_func_factory')