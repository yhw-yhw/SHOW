import mediapipe as mp
import numpy as np
import PIL
import cv2
from glob import glob
import os
from pathlib import Path
import tqdm
from ..utils.paths import glob_exts_in_path
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class FaceDetector:
    def __init__(self, type='google', device='cpu'):
        self.type = type
        self.detector = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)


    def dense_multi_face(self, image):
        # assert(self.type == 'google','FaceDetector => Wrong type for dense detection!')
        results = self.detector.process(image)

        if results.multi_face_landmarks is None:
            return None,None

        lmks_list=[]
        for i in results.multi_face_landmarks:
            lmks = i.landmark
            lmks = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks)))
            lmks[:, 0] = lmks[:, 0] * image.shape[1]
            lmks[:, 1] = lmks[:, 1] * image.shape[0]
            lmks_list.append(lmks)
            
        # lmks_list: (num,478,2)
        return np.array(lmks_list),results
    
    
    def predict_batch(self,
                      img_folder,
                      savefolder,
                      visfolder,
                      saveVis=True):
        Path(savefolder).mkdir(exist_ok=True,parents=True)
        Path(visfolder).mkdir(exist_ok=True,parents=True)
        
        self.imagepath_list = glob_exts_in_path(img_folder,img_ext=['png', 'jpg','jpeg'])
            
        for imagepath in tqdm.tqdm(self.imagepath_list):
            imagename=Path(imagepath).stem
            out_npz_name=os.path.join(savefolder,imagename+'_dense.npz')
            out_img_name=os.path.join(visfolder,imagename+'.jpg')
            self.predict(
                imagepath,
                out_npz_name=out_npz_name,
                out_img_name=out_img_name,
                saveVis=saveVis
            )
                

    def predict(self,full_file_name,out_npz_name,out_img_name,saveVis):
        pil_im = PIL.Image.open(full_file_name).convert('RGB')
        image = np.array(pil_im)
        
        lmks_list,results=self.dense_multi_face(image)
        if lmks_list is not None:
            np.savez(out_npz_name, lmks_list)
            if saveVis:
                image=self.draw(image,results)
                cv2.imwrite(out_img_name, image)
        else:
            open(out_npz_name+'.empty', 'a').close()
            
            
    def draw(self,image,results):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        return image