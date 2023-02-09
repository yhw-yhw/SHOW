
betas_precompute=dict(
    use_pre_compute_betas= True,
    interpenetration= False,
    use_mp_loss= False,
    use_mica_shape= False,
    
    use_height_constraint= False,
    use_weight_constraint= False,
    
    op_shoulder_conf_weight=1.0,
    op_root_conf_weight=0.5,
    
    use_silhouette_loss=False,
    # use_silhouette_loss=True,
    
    use_head_loss= True,
    # use_head_loss= False,
)

pre_compute_betas_weight=dict(
    deca_inner_weight = [0, 1.5, 1.5, 1.5],
    betas_en = [0, 0, 0, 0],
    mica_weight = [0, 0, 0, 0],
    body_weight_weight = [0, 0, 0, 0],
    body_height_weight = [0, 0, 0, 0],
)
        