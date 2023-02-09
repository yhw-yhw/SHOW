import mmcv
from mmcv.utils import Registry
from loguru import logger

MMYLOGGER = Registry('mylogger')


class batch_logger:
    def __init__(self, loggers=[]):
        self.loggers = loggers
        self.warn_filter_list = []

    def __iter__(self):
        pass
    
    def __getstate__(self):
        pass

    def __getattr__(self, name):
        # print(f'__getattr__：{name}')
        func_list = []
        for mylogger in self.loggers:
            mylogger_attr = getattr(mylogger, name, None)

            if mylogger_attr is not None:
                if not callable(mylogger_attr):
                    if name not in self.warn_filter_list:
                        logger.warning(f'{mylogger.__class__.__name__} has attr {name} not callable,'
                                        'you shoud use "batch_logger.loggers[idx].attr" to get attr'
                                        )
                        self.warn_filter_list.append(name)
                    return mylogger_attr
                else:
                    func_list.append(mylogger_attr)
            else:
                if name not in self.warn_filter_list:
                    logger.warning(
                        f'{mylogger.__class__} has no func or attr {name}')
                    self.warn_filter_list.append(name)

        def merge_func(*args, **kwargs):
            for func in func_list:
                func(*args, **kwargs)
        return merge_func


@logger.catch
def build_my_logger(log_config, init_run=True):
    logger_list = []
    log_interval = log_config['interval']
    for info in log_config['hooks']:
        logger_hook = mmcv.build_from_cfg(
            info, MMYLOGGER, default_args=dict(interval=log_interval))
        if init_run:
            # logger_hook.before_run(None)
            logger_hook.initialize()
        logger_list.append(logger_hook)

    
    # return logger_list
    # from ..utils.decorator import batch_logger
    return batch_logger(logger_list)
