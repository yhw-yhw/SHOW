flame_cfg=dict(
    flame_geom_path = '{{ fileDirname }}/../../../data/generic_model.pkl',
    flame_template_path = '{{ fileDirname }}/../../../data/uv_template.obj',
    flame_static_lmk_path = '{{ fileDirname }}/../../../data/flame_static_embedding_68_v4.npz',
    flame_dynamic_lmk_path = '{{ fileDirname }}/../../../data/flame_dynamic_embedding.npy',
    tex_space_path = '{{ fileDirname }}/../../../data/FLAME_albedo_from_BFM.npz',
    num_shape_params = 300,
    num_exp_params = 50,
    tex_params = 150,
    image_size = [512, 512],
)