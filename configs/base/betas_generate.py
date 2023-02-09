
betas_generate=dict(
    use_pre_compute_betas= False,
    interpenetration= False,
    use_mp_loss= False,
    use_mica_shape= True,
    
    # use_height_constraint= True,
    # use_weight_constraint= True, 
    use_height_constraint= False,
    use_weight_constraint= False, 
    
    op_shoulder_conf_weight=0.5,
    op_root_conf_weight=0.5,
    
    # 只有在MPI服务器上才开启
    use_silhouette_loss=False,
    # use_silhouette_loss=True,
    
    # 默认关闭，容易导致梯度爆炸，loss nan
    use_head_loss= False,
)