
betas_generate=dict(
    use_pre_compute_betas= False,
    interpenetration= False,
    use_mp_loss= True,
    # use_mp_loss= False,
    use_mica_shape= True,
    
    # use_height_constraint= True,
    # use_weight_constraint= True, 
    use_height_constraint= False,
    use_weight_constraint= False, 
    
    op_shoulder_conf_weight=0.5,
    op_root_conf_weight=0.5,
    
    # use_silhouette_loss=False,
    use_silhouette_loss=True,
    
    use_head_loss= False,
)