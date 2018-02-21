import os
import torch
import numpy as np
from PIL import Image
from skimage.io import imread
import torch.utils.data as data
    
def gray_2_rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

class FastImageFolder(data.Dataset):
    def __init__(self,
                 root = '',
                 img_list = [],
                 transform = None):
        
        self.root = root
        self.img_list = img_list
        self.transforms = transform

    def __getitem__(self, index):
        _ = 1
        try:
            im = imread(os.path.join(self.root,self.img_list[index]))
            # control for grayscale images
            if len(im.shape)==2:
                im = gray_2_rgb(im)
            # control for images with alpha channel or other weird stuff
            if im.shape[2]>3:
                print('Alpha channel image {}'.format(index))
                im = im[:,:,0:3]
            
            # transform image to PIL format for PyTorch transforms
            im = Image.fromarray(im)
            
            if self.transforms is not None:
                im = self.transforms(im)
        except:
            print('Exception triggered with image {}'.format(index))
            im = imread('random.jpg')
            
            if im.shape[2]>3:
                print('Alpha channel image {}'.format(index))
                im = im[:,:,0:3]
                
            # transform image to PIL format for PyTorch transforms
            im = Image.fromarray(im)
            
            if self.transforms is not None:
                im = self.transforms(im)
        return im,_

    def __len__(self):
        return len(self.img_list) 
