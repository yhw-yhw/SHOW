import numpy as np
from mmcv import BaseFileHandler,register_handler,load,dump



@register_handler('npy')
class NpyHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        return np.load(file)
    # 主要是提供了默认的rb模式
    def load_from_path(self, filepath, **kwargs):
        return super(NpyHandler, self).load_from_path(
            filepath, mode='rb', **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        np.save(file, obj)

    # 主要是提供了默认的wb模式
    def dump_to_path(self, obj, filepath, **kwargs):
        super(NpyHandler, self).dump_to_path(
            obj, filepath, mode='wb', **kwargs)

    def dump_to_str(self, obj, **kwargs):
        # 实际上这么写没有意义
        return obj.tobytes()
    

@register_handler('npz')
class NpzHandler(NpyHandler):
    def dump_to_fileobj(self, obj, file, **kwargs):
        np.savez(file, obj)



if __name__ == '__main__':
    arr1 = np.arange(12).reshape((3, 4))
    dump(arr1, 'out.npy')
    data_str = dump(arr1, file_format='npy')
    print(data_str)
    data = load('out.npy')
    print(data)
    
    dump(arr1, 'out.npz')
    data_str = dump(arr1, file_format='npy')
    print(data_str)
    data = load('out.npz')
    print(data)