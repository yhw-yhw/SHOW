import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from scipy.io import loadmat, savemat
import imageio
import cv2
import argparse
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg


# pixie: 3d smplx model,cam param,crop bbox
# deca: 3d face model,cam param,crop bbox
def dict_tensor2npy(tensor_dict):
    npy_dict = {}
    for key, value in tensor_dict.items():
        # print(type(value))
        # if type(value)==dict:
        #     pass
        #     # print('dict')
        #     # npy_dict[key] = dict_tensor2npy(value)
        # el
        if type(value)==torch.Tensor:
            # print('tensor')
            npy_dict[key] = value.detach().cpu().numpy()
            # print(npy_dict)
        else:
            # print('else')
            npy_dict[key] = value
    return npy_dict


def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # check env
    if not torch.cuda.is_available():
        print('CUDA is not available! use CPU instead')
    else:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    # load video for animation sequence
    posedata = TestData(args.posepath, iscrop=args.iscrop,
                        body_detector='rcnn')

    # -- run PIXIE
    pixie_cfg.model.use_tex = args.useTex
    pixie = PIXIE(config=pixie_cfg, device=device)
    visualizer = Visualizer(render_size=args.render_size, config=pixie_cfg,
                            device=device, rasterizer_type=args.rasterizer_type)

    #TODO deca init
    if args.use_deca:
            # if args.deca_path:
        # os.makedirs(os.path.join(savefolder, 'deca'), exist_ok=True)
        # if given deca code path, run deca to get face details, here init deca model
        sys.path.append(args.deca_path)
        from decalib.deca import DECA
        from decalib.utils.config import cfg as deca_cfg
        from decalib.datasets import datasets as deca_datasets

        deca_cfg.model.use_tex = False
        deca_cfg.rasterizer_type = 'standard'
        deca = DECA(config=deca_cfg, device=device)
        test_deca_datasets = deca_datasets.TestData()

    
    for i, batch_list in enumerate(tqdm(posedata, dynamic_ncols=True)):
        if i % 1 == 0:
            if batch_list is None:
                continue

            original_image=None
            pixie_save_data_list={}
            for index,batch in enumerate(batch_list):
                # bbox = batch['bbox']
                data_name = batch['name']
                
                util.move_dict_to_device(batch, device)
                batch['image'] = batch['image'].unsqueeze(0)
                batch['image_hd'] = batch['image_hd'].unsqueeze(0)
                pixie_param_dict = pixie.encode({'body': batch})
                codedict = pixie_param_dict['body']
                
                extra={
                    'batch':batch
                }
                    
                args_dict = vars(args)
                pixie_opdict = pixie.decode(
                    codedict,
                    param_type='body',
                    extra=extra,
                    **args_dict
                    )

                tform = torch.inverse(batch['tform'][None, ...]).transpose(1, 2)
                
                if original_image is None:
                    original_image = batch['original_image'][None, ...]
                elif visdict['color_shape_images'] is not None:
                    original_image=visdict['color_shape_images']
                # original_image = batch['original_image'][None, ...]
                
                visualizer.recover_position(
                    pixie_opdict, batch, tform, original_image)
                visdict = visualizer.render_results(
                    pixie_opdict, batch['image_hd'],
                    moderator_weight=pixie_param_dict['moderator_weight'],
                    overlay=True)
                
                pixie_save_data={
                    'focal':args_dict['focal'],
                    'dataset_batch':batch,
                    'pixie_opdict': dict_tensor2npy(pixie_opdict),
                    'param_dict_axis':dict_tensor2npy(pixie_opdict['param_dict_axis']),
                }
                pixie_save_data_list[f'body{index}']=pixie_save_data

            #TODO img save
            if args.saveVis:
                save_img_path=os.path.join(savefolder, f'{data_name}.jpg')
                cv2.imwrite(
                    save_img_path,
                    visualizer.visualize_grid(
                        {'pose_ref_shape': visdict['color_shape_images'].clone()}, size=512)
                )
                
            #TODO PIXIE save
            batch.pop('image')
            batch.pop('image_hd')
            savemat(
                os.path.join(savefolder, f'{data_name}.mat'), 
                pixie_save_data_list
                )
            
    print(f'-- please check the results in {savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')

    parser.add_argument('--use_deca', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--focal', default=5000, type=int,
                        help='image size of renderings')
    
    
    # parser.add_argument('-i', '--inputpath', default='TestSamples/body/woman-in-white-dress-3830468.jpg', type=str,
    #                     help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-p', '--posepath', default='TestSamples/animation', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    # rendering option
    parser.add_argument('--render_size', default=1024, type=int,
                        help='image size of renderings')
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--reproject_mesh', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to reproject the mesh and render it in original image size, \
                            currently only available if rasterizer_type is standard, because pytorch3d does not support non-squared image...\
                            default is False, means use the cropped image and its corresponding results')
    # save
    parser.add_argument('--deca_path', default='C:\\Users\\lithiumice\\code\\DECA', type=str,
                        help='absolute path of DECA folder, if exists, will return facial details by running DECA\
                        details of DECA: https://github.com/YadiraF/DECA')
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--uvtex_type', default='SMPLX', type=str,
                        help='texture type to save, can be SMPLX or FLAME')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveGif', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to visualize other views of the output, save as gif')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, \
                            Note that saving objs could be slow')
    parser.add_argument('--saveParam', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save parameters as pkl file')
    parser.add_argument('--savePred', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save smplx prediction as pkl file')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    main(parser.parse_args())
