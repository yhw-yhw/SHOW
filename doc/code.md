# Code Structure

## configs

```bash
├─configs
│      base.py
│      condor_mmcv_cfg.py
│      local_machine_cfg.py
│      MPI_machine_cfg.py
│      speaker_info.py
│  cfg_ins.py                       # condor的config
│  condor_batch.py                  # 提交condor jobs
│  condor_entry.py                  # 运行所有task的入口
│  csv_parser.py                    # 输入csv中的某一行，输出对应视频片段的所有路径
│  run.sh
│  task_export.py                   # 导出wav和pkl的tar
│  task_sample.py                   # sample数据集
│  task_smplifyx.py                 # 跑smplifyx算法
│  task_tracker.py                  # 跑face tracking算法
│  template.sh
│  template.sub
│  utils.py
│  _init_paths.py                   # 自动导入core_libs的PATH路径
│  __init__.py
```

## configs

```bash
├─base                              # 配置config的基本组件
│      face_ider.py                 # 人脸匹配设置
│      log_config.py                # logger的配置
│      model_flame_config.py        # FLAME的模型路径配置
│      model_smplx_config.py        # SMPLX的模型路径配置
│      optional_path_oliver.py      # smplifyx的文件路径
│      optimizer_config.py          # 优化器参数设置
│      smplifyx_loss_configs.py     # smplifyx的loss开关
│      smplifyx_prior_config.py     # smplifyx的prior设置
│      smplifyx_weights.py          # smplifyx的权重设置
│
├─smplifyx_config
│      betas_generate.py
│      betas_precompute.py
│
├─yaml_configs
│  │  smplifyx_configparser.py
│  │  smplifyx_yacs_config.py
│  │  tracker_config.py
│  │
│  ├─face_tracker
│  │      flame_tracker.yml
│  │      smplx_tracker.yml
│  │
│  └─smplifyx_video
│          betas_generate.yml
│          use_pre_betas.yml
│  mmcv_smplifyx_config.py          # smplifyx的所有设置
│  mmcv_tracker_config.py           # tracking的所有设置
```

## moduels

```bash
├─SHOW
│  ├─datasets
│     op_dataset.py 
│     op_post_process.py    
│     pre_dataset.py                # 加载PIXIE、DECA、mediapipe等预先运行的结果
│  ├─face_iders
│  │  builder.py                    # 使用‘IDER’注册两个face identitfyer
│  │  arcface_ider.py               # 使用arcface pytorch作为face identitfyer
│  │  insightface_ider.py           # 使用insightface作为face identitfyer
│  ├─flame
│  │  FLAME.py
│  │  lbs.py
│  ├─loggers
│  │  └─__pycache__
│  ├─utils
│  │  colab.py
│  │  decorator.py
│  │  disp_img.py
│  │  fun_factory.py                # 类似MMCV中的REGISTER，注册字符串到函数的map
│  │  metric.py                     
│  │  misc.py
│  │  npy_handler.py
│  │  op_utils.py
│  │  paths.py
│  │  render.py
│  │  timer.py
│  │  video.py
│  ├─video_filter
│  face_detector.py                 # 使用FAN和mediapipe并保存结果为pkl
│  image.py
│  kinect.py
│  lbfgs_ls.py
│  load_assets.py                   # 加载data下的文件
│  load_models.py                   # 加载SMPLX model和vposer model
│  losses.py
│  masking.py
│  parse_weight.py
│  pifpaf_detector.py
│  prior.py
│  renderer.py
│  save_results.py
│  smplx_dataset.py
│  tracker_rasterizer.py
│  util.py
│
├─requirements
├─tools
├─transformations
```

## models and data file tree

```bash
├─data
│  └─id_pic
│      ├─arcface
│      └─insightface
├─models
│  ├─arcface
│  │  └─glink360k_cosface_r100_fp16_0.1.pth
│  │  └─ms1mv3_arcface_r50_fp16.pth
│  ├─models_deca
│  │  └─data--deca_model.tar
│  ├─models_pixie
│  │  └─data--pixie_model.tar
│  ├─smplx
│  ├─models_LoFTR
│  ├─homogenus_v1_0
│  └─vposer_v1_0
├─modules
   ├─arcface_torch
   ├─COAP
   ├─DECA
   ├─LEMO
   ├─LoFTR
   ├─mesh-isect
   ├─MICA
   ├─MP
   └─PIXIE
```
