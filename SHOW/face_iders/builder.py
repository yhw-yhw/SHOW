from mmcv.utils import Registry
import mmcv

IDER = Registry('ider')

def build_ider(config):
    return mmcv.build_from_cfg(config,IDER)

def build_ider2(config):
    return IDER.build(config)