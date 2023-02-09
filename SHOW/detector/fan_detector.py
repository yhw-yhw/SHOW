from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import SHOW
import cv2
import torch
import torchvision.transforms.functional as F
from PIL import Image


class FAN_Detector(object):
    def __init__(self,device='cuda'):
        if self.__dict__.get('face_detector',None) is None:
            import face_alignment
            self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
        
    def predict(self,
                img_folder,fan_npy_folder,
                save_vis,fan_vis_dir):
        if save_vis:
            os.makedirs(fan_vis_dir,exist_ok=True)

        Path(fan_npy_folder).mkdir(exist_ok=True, parents=True)
        imagepath_list = SHOW.glob_exts_in_path(
            img_folder,
            img_ext=['png', 'jpg', 'jpeg'])

        for imagepath in tqdm(imagepath_list):
            imagename = Path(imagepath).stem
            lmk_path = os.path.join(fan_npy_folder, imagename + '.npy')
            
            if not os.path.exists(lmk_path):
                
                # cv2_image = cv2.imread(imagepath)
                pil_image = Image.open(imagepath).convert("RGB")
                # orig_width, orig_height = pil_image.size
                image_np = np.array(pil_image)
                
                lmks = self.face_detector.get_landmarks(image_np)  # list

                if lmks is not None:
                    lmks = np.array(lmks)
                    np.save(lmk_path, lmks)
                else:
                    open(
                        os.path.join(
                            fan_npy_folder, 
                            imagename + '.npy.empty'),
                        'a'
                    ).close()
                    
                if save_vis:
                    image_torch = F.to_tensor(pil_image)
                    ret_img = SHOW.tensor_vis_landmarks(
                        image_torch,
                        torch.from_numpy(lmks[0])
                    )
                    m_img = SHOW.merge_views([[ret_img[0]]])
                    cv2.imwrite(f'{fan_vis_dir}/{imagename}.jpg', m_img)
                    
        