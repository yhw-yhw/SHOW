
import os
import cv2
import numpy as np
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map
from torchvision.transforms.functional import gaussian_blur
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import PIL
import time

def img_preprocess(img):
    # img: 0-1
    # img: tensor or ndarray
    # img: (1,c,h,w)
    # img: (c,h,w)
    # img: (h,w,c)
    # img: cpu or gpu
    # img: frad or no_grad
    if isinstance(img,torch.Tensor):
        img=img.cpu().detach()
        if img.ndimension()==4:
            img=img[0]
        if img.shape[0]==3 or img.shape[0]==1:
            img=img.permute(1,2,0)
            
    if isinstance(img,np.ndarray):
        if img.ndim==4:
            img=img[0]
        if img.shape[0]==3 or img.shape[0]==1:
            img=img.transpose(1,2,0)
    # return: (h,w,3);0-1
    return img

def show_PIL_im_window(tensor):
    import PIL
    
    img=img_preprocess(tensor)
    
    if isinstance(img,torch.Tensor):
        img=img.numpy()
        
    scale=255/img.max()
    img = (img.copy() * scale)
    if img.shape[-1]==3:
        img=img[:, :, [2, 1, 0]]
    if img.shape[-1]==1:
        img=np.repeat(img,3,-1)
    img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
    PIL.Image.fromarray(img).show()

    
def save_tensor_to_file(tensor, path='tensor.jpg'):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor[0].detach().cpu().numpy()
    img = (tensor.transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
    cv2.imwrite(path, img)


def show_plt_fig_face(datas):
    plt.figure()
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.axis('equal')
    plt.scatter(datas[:, 0], datas[:, 1])
    for i, p in enumerate(datas):
        plt.annotate(str(i), p)
    plt.show()


def show_plt_fig_im(X):
    X = X[0].detach().cpu().numpy()*255
    X = X.astype(np.uint8).transpose(1, 2, 0)
    plt.imshow(X)



def gen_cheers_board(
    im_size = 8,
    im_size2 = 8
):

    a=np.zeros([im_size,im_size2])
    b=np.zeros([im_size,im_size2])

    for i in range(a.shape[0]):
        if i%2==0:
            a[i,:]=1
        else:
            b[:,i]=1

    im_data=np.ones([im_size,im_size2])*np.logical_xor(a,b)
    return im_data


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image
