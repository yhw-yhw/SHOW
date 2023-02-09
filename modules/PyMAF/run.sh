

image_folder=$all_top_dir/image
pymaf_folder=$all_top_dir/pymaf
python -m apps.demo_smplx \
--image_folder $image_folder \
--output_folder $pymaf_folder \
--detection_threshold 0.3 \
--pretrained_model data/pretrained_model/PyMAF-X_model_checkpoint.pt \
--misc TRAIN.BHF_MODE full_body MODEL.EVAL_MODE True MODEL.PyMAF.HAND_VIS_TH 0.1
