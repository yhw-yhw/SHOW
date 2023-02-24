import torch
import cv2, tqdm
import numpy as np
import torch, torchvision
import matplotlib.cm as cm
from torchvision import transforms
from multiprocessing import Pool
import os.path as osp
from PIL import Image
# import trimesh,pyrender
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3
from mpl_toolkits.mplot3d import Axes3D
import shutil, argparse, subprocess, time, json, glob, pickle, yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from dataclasses import dataclass
import os
from pathlib import Path
import tqdm
from ..utils.paths import glob_exts_in_path
from scipy.ndimage.morphology import distance_transform_edt


class deeplab_seg(object):

    def __init__(self):
        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.deeplab_model = torch.hub.load('pytorch/vision:v0.6.0',
                                            'deeplabv3_resnet101',
                                            pretrained=True).to(self.device)
        self.deeplab_model.eval()

        self.deeplab_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict_batch(self, img_folder, savefolder, saveVis=True, **kwargs):
        Path(savefolder).mkdir(exist_ok=True, parents=True)
        self.imagepath_list = glob_exts_in_path(img_folder,
                                                img_ext=['png', 'jpg', 'jpeg'])

        for imagepath in tqdm.tqdm(self.imagepath_list):
            imagename = Path(imagepath).stem
            vis_path = os.path.join(savefolder, imagename + '.jpg')
            self.predict(im_path=imagepath,
                         vis_path=vis_path if saveVis else None,
                         **kwargs)


    def predict(self, im_path, vis_path=None, save_mode='bin'):
        sample_img = cv2.imread(im_path)
        input_tensor = self.deeplab_preprocess(sample_img)
        input_tensor = input_tensor.to(self.device).unsqueeze_(dim=0)

        with torch.no_grad():
            output = self.deeplab_model(input_tensor)['out']

        self.person_seg = torch.logical_not(output.argmax(1) == 15).to(
            torch.float)

        if vis_path is not None:
            if save_mode == 'crop':
                self.save_crop_out(self.person_seg, vis_path, sample_img)
            elif save_mode == 'mask':
                self.save_org_mask(self.person_seg, vis_path, sample_img)
            elif save_mode == 'bin':
                self.save_mask(self.person_seg, vis_path)

    def save_org_mask(self,
                      person_seg_org,
                      vis_path,
                      sample_img,
                      color=(0, 0, 255),
                      opacity=0.5):

        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

        if isinstance(person_seg_org, torch.Tensor):
            person_seg = person_seg_org.cpu().numpy()

        person_seg = person_seg.transpose(1, 2, 0)
        person_seg = (1 - person_seg)

        color_seg = np.zeros((person_seg.shape[0], person_seg.shape[1], 3),
                             dtype=np.uint8)
        color_seg[person_seg[:, :, 0] == 1, :] = color
        color_seg = color_seg[..., ::-1]
        # convert to BGR

        sample_img[person_seg[:,:,0] == 1, :]=\
            sample_img[person_seg[:,:,0] == 1, :]*(1-opacity)
        sample_img = sample_img + (opacity * color_seg)

        out_img = Image.fromarray(sample_img.astype(np.uint8))
        out_img.save(vis_path)

    def save_crop_out(self, person_seg_org, vis_path, sample_img):
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

        if isinstance(person_seg_org, torch.Tensor):
            person_seg = person_seg_org.cpu().numpy()

        person_seg = person_seg.transpose(1, 2, 0)
        sample_img = sample_img * (1 - person_seg)
        sample_img = sample_img + person_seg * 255
        out_img = Image.fromarray(sample_img.astype(np.uint8))
        out_img.save(vis_path)

    def save_mask(self, person_seg, vis_path):
        if isinstance(person_seg, torch.Tensor):
            person_seg = person_seg.cpu().numpy()

        person_mask0 = (person_seg[0] * 255.0).astype(np.uint8)
        out_img = Image.fromarray(person_mask0)
        out_img.save(vis_path)

    def deeplab_video(self,sample_img_list,predict_size):
        img_batch = torch.zeros((len(sample_img_list), 3, predict_size[1], predict_size[0]))
        
        for i,sample_img in enumerate(sample_img_list):
            sample_img=cv2.resize(sample_img,predict_size)
            input_tensor = self.deeplab_preprocess(sample_img)
            img_batch[i] = input_tensor
            img_batch = img_batch.to(self.device)
            
        with torch.no_grad():
            output = self.deeplab_model(img_batch)['out']
            
        self.person_seg = torch.logical_not(output.argmax(1) == 15).to(torch.float)
        self.person_seg = self.person_seg.cpu().numpy()
        
        return self.person_seg
