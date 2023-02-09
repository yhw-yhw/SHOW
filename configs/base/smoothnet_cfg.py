cfg_smoothnet_w8 = dict(
    type='smoothnet',
    window_size=8,
    output_size=8,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/../models/smoothnet/smoothnet_windowsize8.pth.tar?versionId'
    '=CAEQPhiBgMDo0s7shhgiIDgzNTRmNWM2ZWEzYTQyYzRhNzUwYTkzZWZkMmU5MWEw',
    device='cpu')

cfg_smoothnet_w16 = dict(
    type='smoothnet',
    window_size=16,
    output_size=16,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/../models/smoothnet/smoothnet_windowsize16.pth.tar?versionId'
    '=CAEQPhiBgMC.s87shhgiIGM3ZTI1ZGY1Y2NhNDQ2YzRiNmEyOGZhY2VjYWFiN2Zi',
    device='cpu')

cfg_smoothnet_w32 = dict(
    type='smoothnet',
    window_size=32,
    output_size=32,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/../models/smoothnet/smoothnet_windowsize32.pth.tar?versionId'
    '=CAEQPhiBgIDf0s7shhgiIDhmYmM3YWQ0ZGI3NjRmZTc4NTk2NDE1MTA2MTUyMGRm',
    device='cpu')

cfg_smoothnet_w64 = dict(
    type='smoothnet',
    window_size=64,
    output_size=64,
    checkpoint='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
    'mmhuman3d/../models/smoothnet/smoothnet_windowsize64.pth.tar?versionId'
    '=CAEQPhiBgMCyw87shhgiIGEwODI4ZjdiYmFkYTQ0NzZiNDVkODk3MDBlYzE1Y2Rh',
    device='cpu')