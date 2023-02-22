from functools import partial,reduce
import platform
import os
import torch
import numpy as np
import re

def replace_spec_code(in_name):
        
    disp_name=re.sub(
        re.compile(
            r'[^a-zA-Z0-9]'
            # r'[-,$()#+&*]'
            ),
        "_",
        in_name)
    
    return disp_name

def cvt_dict_to_tensor(data,device,dtype):
    if isinstance(data,list):
        return [cvt_dict_to_tensor(v,device,dtype) for v in data]
    elif isinstance(data,dict):
        return {k : (cvt_dict_to_tensor(v,device,dtype) if k!='seg_stack' else v)
                for k,v in data.items()}
    else:
        if isinstance(data,np.ndarray):
            return torch.from_numpy(data).to(device=device,dtype=dtype)
        elif isinstance(data,torch.Tensor):
            return data.to(device=device,dtype=dtype)
        else:
            raise ValueError(f'not support type {type(data)}')

def expand_var_shape(var,target_len=300,expand_axis=-1):
    if isinstance(var,np.ndarray):
        org_len=var.shape[expand_axis]
        if target_len==org_len:
            return
        oth_len=var.shape[:expand_axis]
        new_var=np.concatenate([var,np.zeros(*oth_len,target_len-org_len)],axis=expand_axis)
        return new_var
    if isinstance(var,torch.Tensor):
        org_len=var.shape[expand_axis]
        if target_len==org_len:
            return
        oth_len=var.shape[:expand_axis]
        new_var=torch.cat([var,torch.zeros(*oth_len,target_len-org_len)],axis=expand_axis)
        return new_var
    
    
def str_to_torch_dtype(s):
    dtype = torch.float32
    if s == 'float64':
        dtype = torch.float64
    elif s == 'float32':
        dtype = torch.float32
    return dtype

def reload_module(s:str='SHOW.smplx_dataset'):
    import imp
    eval(f'import {s} as target')
    imp.reload(locals()['target'])

def print_args(args:dict):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")

def print_dict_losses(losses):
    return reduce(
        lambda a, b: a + f' {b}={round(losses[b].item(), 4)}', 
        [""] + list(losses.keys())
    )

def platform_init():
    import platform
    if platform.system() == "Linux":
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
    else:
        if 'PYOPENGL_PLATFORM' in os.environ:
            os.environ.__delitem__('PYOPENGL_PLATFORM')
        
def work_seek_init(rank = 42):
    import torch
    import torch.backends.cudnn
    import numpy as np
    
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(rank)
    
    torch.backends.cudnn.enabled = False


    
def get_gpu_info():
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count= pynvml.nvmlDeviceGetCount()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name=pynvml.nvmlDeviceGetName(handle).decode()
        gpu_version=pynvml.nvmlSystemGetDriverVersion()
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        B_to_MB=1024*1024
        gpu_Total = info.total/B_to_MB
        gpu_Free = info.free /B_to_MB
        gpu_Used = info.used /B_to_MB
        
        return dict(
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_version=gpu_version,
            gpu_Total=gpu_Total,
            gpu_Free=gpu_Free,
            gpu_Used=gpu_Used
        )
    except:
        import traceback
        traceback.print_exc()
        

def get_machine_info():
    host_name=platform.node()
    gpu_info = get_gpu_info()
    
    machine_info=dict(
        host_name=host_name,
        gpu_info=gpu_info,
    )
    return machine_info