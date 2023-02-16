log_config_tracker_nep = dict( 
    interval=100, 
    hooks=[ 
        dict(type='MyNeptuneLogger', 
             with_step=False,
             init_kwargs=dict(
                 name='test',
                 project='lithiumice/tracker',
                 api_token="==",
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
                 api_token="==",
                 )) 
    ])  

log_config_smplifyx_wandb = dict( 
    interval=100, 
    hooks=[ 
        dict(type='MyWandbLogger', 
             log_artifact=False,
             wandb_key='',
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
             wandb_key='',
             wandb_name='NEED_TO_BE_FILLED',
             init_kwargs=dict(
                 reinit=True,
                 resume='allow',
                 project='tracker',
                 )),
    ])  
