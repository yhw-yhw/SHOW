import cv2


class MMPoseAnalyzer():
    def __init__(self):
        from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                        vis_pose_result, process_mmdet_results)
        from mmdet.apis import inference_detector, init_detector
        import os
        mmpose_root = os.environ.get('mmpose_root')
        pose_config = os.path.join(mmpose_root,'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py')
        det_config = os.path.join(mmpose_root,'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py')
        
        pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

        self.pose_model = init_pose_model(pose_config, pose_checkpoint)
        self.det_model = init_detector(det_config, det_checkpoint)

    def predict(self, img):
        from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                        vis_pose_result, process_mmdet_results)
        from mmdet.apis import inference_detector, init_detector

        mmdet_results = inference_detector(self.det_model, img)
        person_results = process_mmdet_results(mmdet_results, cat_id=1)
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results,
            bbox_thr=0.3,
            format='xyxy',
            dataset=self.pose_model.cfg.data.test.type)
        
        # # (left, top, right, bottom, [score])

        return pose_results
        
    def visualize(self,img,pose_results):
        from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                        vis_pose_result, process_mmdet_results)
        from mmdet.apis import inference_detector, init_detector

        vis_result = vis_pose_result(
            self.pose_model,
            img,
            pose_results,
            dataset=self.pose_model.cfg.data.test.type,
            show=False)
        vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)
        return vis_result
        
    
            
