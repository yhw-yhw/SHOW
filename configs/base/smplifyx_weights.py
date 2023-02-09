
opt_weights_dict = dict(
    jaw_prior_weight=[[47.8, 478.0, 478.0],]*4,
    bending_prior_weight=[15.15,]*4,
    body_pose_weight=[4.78, ]*4,
    expr_prior_weight=[5.0, ]*4,
    hand_prior_weight=[4.78,]*4,
    coll_loss_weight=[1.0,]*4,
    shape_weight=[5.0,]*4,
    data_weight=[1.0,]*4,

    body_joints_weight=[2,],
    hand_joints_weight=[0, 1.5, 1.5, 1.5],
    
    selfpen_weight=[0, 0, 0, 3.0],
    w_deg_range=[20, 20, 20, 20],
    w_s2d_body=[0.1, 0.1, 0.1, 0.1],
    w_s2d_hand=[0, 0, 0, 0],
    w_s3d_body=[1, 5.5, 7.5, 7.5],
    w_s3d_hand=[0, 1, 1, 1],
    w_svposer=[10, 120, 130, 150],
    w_spca_hand=[0, 2, 5, 5],
    
    wl_silhouette=[0,0,0,8.0],

    w_deca_outter=[10.0, 10.0, 10.0, 10.0],
    w_deca_inner=[0, 0.5, 0.5, 0.5],

    mp_weight=[0, 2.5, 2.5, 2.5],
    mica_weight=[0, 0.5, 0.5, 0],

    w_body_weight=[6, 6, 6, 0],
    w_body_height=[100, 40, 40, 0],
    
    w_transl_smooth=[5.0, 0, 0, 0],
    w_orient_smooth=[1.0, 1.0, 0, 0],
    
    expression_en=[0, 0, 1, 1],
    jaw_pose_en=[1, 1, 1, 1],
    betas_en=[1, 1, 1, 0],
    transl_en=[1, 0, 0, 0],
    global_orient_en=[1, 1, 0, 0],
    pose_embedding_en=[1, 1, 1, 1],
    
    left_hand_pose_en=[0, 1, 1, 1],
    right_hand_pose_en=[0, 1, 1, 1],
    leye_pose_en=[0, 1, 1, 1],
    reye_pose_en=[0, 1, 1, 1],
    mica_head_transl_en=[0, 1, 0, 0],
    mica_en=[0, 1, 0, 0],

)
