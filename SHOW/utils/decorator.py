import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from loguru import logger


def try_statement(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            import traceback
            traceback.print_exc()
    return wrapper    


# convert a function into recursive style to 
# handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)
    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float) or isinstance(vars, int):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

@make_recursive_func
def to_tensor(vars):
    if isinstance(vars, np.ndarray):
        return torch.from_numpy(vars)
    elif isinstance(vars, (int,float)):
        return torch.tensor(vars)
    else:
        return vars
    
@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))
