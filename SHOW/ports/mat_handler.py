import numpy as np
from mmcv import BaseFileHandler,register_handler,load,dump
from scipy.io import loadmat,savemat


@register_handler('mat')
class MatHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        return loadmat(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        savemat(file, obj)
        
        
    def load_from_path(self, filepath, **kwargs):
        return super().load_from_path(
            filepath, mode='rb', **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super().dump_to_path(
            obj, filepath, mode='wb', **kwargs)

    def dump_to_str(self, obj, **kwargs):
        # 实际上这么写没有意义
        return obj.tobytes()

if __name__ == '__main__':
    arr1 = np.arange(12).reshape((3, 4))
    dump(arr1, 'out.mat')
    data_str = dump(arr1, file_format='mat')
    print(data_str)
    data = load('out.mat')
    print(data)
    