import os, sys
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob

from ..utils import util
from . import detectors
from ..utils import array_cropper

def build_dataloader(testpath, batch_size=1):
    data_list = []
    dataset = TestData(testpath = testpath)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last =False)
    return dataset, dataloader

def video2sequence(video_path):
    print('extract frames from video: {}...'.format(video_path))
    videofolder = video_path.split('.')[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        success,image = vidcap.read()
        if image is None:
            break
        if count % 1 == 0:
            imagepath = '{}/{}_frame{:05d}.jpg'.format(videofolder, video_name, count)
            cv2.imwrite(imagepath, image)     # save frame as JPEG file
            imagepath_list.append(imagepath)
        count += 1

    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData(Dataset):
    def __init__(self, testpath, iscrop=False, 
                 crop_size=224, hd_size = 1024, 
                 scale=1.1, body_detector='rcnn', 
                 device='cpu',
                 en_multi_person=True
                 ):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        def glob_exts_in_path(path, img_ext=['png', 'jpg']):
            from functools import reduce
            return reduce(
                lambda before, ext: before+glob(
                    os.path.join(path, f'*.{ext}')
                ),
                [[]]+img_ext)
            
        if os.path.isdir(testpath):
            self.imagepath_list=glob_exts_in_path(testpath)
            # self.imagepath_list=[os.path.join(testpath, i) for i in os.listdir(testpath)]
            # self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.jpeg')
        elif isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'MOV']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print(f'please check the input path: {testpath}')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.hd_size = hd_size
        self.scale = scale
        self.iscrop = iscrop
        self.en_multi_person=en_multi_person
        
        if body_detector == 'rcnn':
            self.detector = detectors.FasterRCNN(device=device)
        elif body_detector == 'keypoint':
            self.detector = detectors.KeypointRCNN(device=device)
        elif body_detector == 'mmdet':
            self.detector = detectors.MMDetection(device=device)
        else:
            print('no detector is used')

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        # print(f"getting {index}")
        imagepath = self.imagepath_list[index]
        
        imagename=os.path.split(imagepath)[-1]
        imagename=os.path.splitext(imagename)[0]
                
        image = imread(imagepath)[:,:,:3]/255.
        h, w, _ = image.shape
        org_img_height=h
        org_img_width=w

        image_tensor = torch.tensor(image.transpose(2,0,1), dtype=torch.float32)[None, ...]
        # if self.iscrop:
        
        prediction = self.detector.run(image_tensor)
        inds = (prediction['labels']==1)*(prediction['scores']>0.5)
        
        
        if inds.count_nonzero()==0:
            print('no person detected! run original image')
            return {'name': imagename,'is_missing':True}
        
        #################################################
        person_boxes=prediction['boxes'][inds]
        return_item_list=[]
        # #############
        # box_size_list=[]
        # for person_boxe in person_boxes:
        #     box_size=(person_boxe[2]-person_boxe[0])*(person_boxe[3]-person_boxe[1])
        #     box_size_list.append(box_size)
        
        # max_size_person_idx=box_size_list.index(max(box_size_list))
        # bbox = person_boxes[max_size_person_idx].cpu().numpy()
        # #############
        for bbox in person_boxes:
            bbox=bbox.cpu().numpy()
                
            left = bbox[0]; right = bbox[2]; top = bbox[1]; bottom = bbox[3]
            
                
            old_size = max(right - left, bottom - top)
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*self.scale)
            src_pts = np.array([
                [center[0]-size/2, center[1]-size/2], 
                [center[0] - size/2, center[1]+size/2], 
                [center[0]+size/2, center[1]-size/2]])
            # print(f'size:{size}')
            new_bbox=[center[0]-size/2, center[1]-size/2,center[0] + size/2, center[1]+size/2]
            # print(f'new_bbox:{new_bbox}')
                
                
            # else:
            #     src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
            #     left = 0; right = w-1; top=0; bottom=h-1
            #     bbox = [left, top, right, bottom]
            #     new_bbox=bbox

            # crop image
            DST_PTS = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
            dst_image = dst_image.transpose(2,0,1)
            # a=tform.param
            # a.dot(src_pts) --> DST_PTS

            DST_PTS = np.array([[0,0], [0,self.hd_size - 1], [self.hd_size - 1, 0]])
            tform_hd = estimate_transform('similarity', src_pts, DST_PTS)
            hd_image = warp(image, tform_hd.inverse, output_shape=(self.hd_size, self.hd_size))
            hd_image = hd_image.transpose(2,0,1)
            
            # tform*[int(left),int(top),1]
            # d=[right,bottom,1]

            return_item= {
                'image': torch.tensor(dst_image).float(),
                'name': imagename,
                'imagepath': imagepath,
                'image_hd': torch.tensor(hd_image).float(),
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                'bbox': new_bbox,
                'size': size,
                'org_img_size':[org_img_height,org_img_width]
                # 'is_person_deted':is_person_deted
                }
            return_item_list.append(return_item)
        return return_item_list
