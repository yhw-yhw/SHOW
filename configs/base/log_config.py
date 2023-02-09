log_config_tracker_nep = dict( 
    interval=100, 
    hooks=[ 
        dict(type='MyNeptuneLogger', 
             with_step=False,
             init_kwargs=dict(
                 name='test',
                 project='lithiumice/tracker',
                 api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYzVkZmE4MC1hYzI4LTQ1ZDYtYjY4Yy1hY2RhM2MwMzY0Y2UifQ==",
                 )) 
    ])  

log_config_smplifyx_nep = dict( 
    interval=100, 
    hooks=[ 
        dict(type='MyNeptuneLogger', 
             with_step=False,
             init_kwargs=dict(
                 name='test',
                 project='lithiumice/smplifyx',
                 api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYzVkZmE4MC1hYzI4LTQ1ZDYtYjY4Yy1hY2RhM2MwMzY0Y2UifQ==",
                 )) 
    ])  

log_config_smplifyx_wandb = dict( 
    interval=100, 
    hooks=[ 
        dict(type='MyWandbLogger', 
             log_artifact=False,
             wandb_key='e3d537403fce5c8a99893c2cbe20a8d49a79358d',
             wandb_name='NEED_TO_BE_FILLED',
             init_kwargs=dict(
                 reinit=True,
                 resume='allow',
                 project='smplifyx',
                 )),
    ])  

log_config_tracker_wandb = dict( 
    interval=100, 
    hooks=[ 
        dict(type='MyWandbLogger', 
             log_artifact=False,
             wandb_key='e3d537403fce5c8a99893c2cbe20a8d49a79358d',
             wandb_name='NEED_TO_BE_FILLED',
             init_kwargs=dict(
                 reinit=True,
                 resume='allow',
                 project='tracker',
                 )),
    ])  
