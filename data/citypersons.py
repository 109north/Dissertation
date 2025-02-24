#!/usr/bin/env python
# coding: utf-8

# In[149]:


import glob
import os
import random
import pandas as pd
import cv2
import warnings

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

def load_images_and_anns(im_dir, ann_file):
    r"""
    Method to get the csv file and for each line
    get all the objects and their ground truth detection
    information for the dataset
    :param im_dir: Path of the images
    :param ann_file: Path of annotation csv file
    :param label2idx: Class Name to index mapping for dataset
    :return:
    """
    # YOU SHOULD PUT THIS OUTSIDE THE FUNCTION 
    ann_df = pd.read_csv(ann_file)
    ann_df = ann_df.groupby('image_id').agg({ # pivot the dataframe so that there's one line per image
        'Unnamed: 0': list,
        'city': 'first',  # Keep only the first value (since it's the same for each image)
        'class_label': list,
        'x1': list,   # Store as list
        'y1': list,
        'w': list,
        'h': list,
        'instance_id': list,
        'x1_vis': list,
        'y1_vis': list,
        'w_vis': list,
        'h_vis': list
    
    }).reset_index().rename(columns={'Unnamed: 0':'box_id'})
    # drop the rows which image doesn't exist on my computer
    # *THIS STEP NOT NECESSARY SINCE IVE DROPPED ALL ENTRIES ORIGINALLY LABELED 'IGNORE REGION'*
    #dfls = list(ann_df['image_id']) # list of image ids in the dataframe
    #ls = os.listdir('citypersons_dir/train/images') # list of image ids on my computer
    #drop = list(set(dfls) - set(ls)) # entries that aren't on the computer
    #missing = [item in drop for item in ann_df['image_id'].to_list()]
    #ann_df = ann_df.loc[~pd.Series(missing, index=ann_df.index)] # drop the corresponding rows
    
    im_infos = []
    for index, row in tqdm(ann_df.iterrows()): # for each line in the annotations dataframe
        im_info = {}
        im_info['img_id'] = row['image_id'].split('_leftImg8bit.png')[0] # get the image id by splitting at the suffix
        im_info['filename'] = os.path.join(im_dir, '{}_leftImg8bit.png'.format(im_info['img_id'])) # creates a path to the relevant image file by adding the png suffix onto the image id from above line
        #image_BGR = cv2.imread(im_info['filename']) # read the image with cv2
        #image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB) # convert to RBG
        #image_tensor = torch.from_numpy(image_RGB).float().permute(2, 0, 1).to("cuda") / 255.0  # Normalize to [0,1]
        im = Image.open(im_info['filename'])
        im_tensor = torchvision.transforms.ToTensor()(im)
        h, w = im_tensor.shape[:2] # get height and width of image
        im_info['width'] = w
        im_info['height'] = h
        detections = []

        #print(im_info['img_id'])
        for i in range(len(list(row['box_id']))): # for each box index in the box_id column
            det = {}
            #label = label2idx[obj.find('name').text]
            #bbox_info = obj.find('bndbox')
            bbox = [
                int(list(row['x1'])[i]),
                int(list(row['y1'])[i]),
                int(list(row['x1'])[i]) + int(list(row['w'])[i]),
                int(list(row['y1'])[i]) + int(list(row['h'])[i])
            ]
            det['label'] = list(row['class_label'])[i] # **FOR NOW IM JUST GOING TO LABEL THEM WITH THE NUMBERS**
            det['bbox'] = bbox
            detections.append(det)
            #print(F"box {i} image {im_info['img_id']}")

        im_info['detections'] = detections
        im_infos.append(im_info)
        
        if self.split=='train' and args.flip: #if --flip=True in CLI arguments in training script, and only if we are in training
            if random.random() < args.flip_percent: #if random value between 0 and 1 is within our flip percent
                flipped_im_info = im_info.copy() 
                flipped_im_info['img_id'] += '_flipped' #change the image id
                flipped_im_info['filename'] = None  # No actual file, just augment in memory
                flipped_detections = []

                for det in detections:
                    # flip the bbox
                    x1, y1, x2, y2 = det['bbox']
                    bbox_width = x2 - x1 
                    flipped_x1 = w - x2
                    flipped_x2 = flipped_x1 + bbox_width
                    flipped_detections.append({'label': det['label'], 'bbox': [flipped_x1, y1, flipped_x2, y2]})
            
                flipped_im_info['detections'] = flipped_detections
                im_infos.append(flipped_im_info) # append the flipped info into all the image infos
    `    
        if self.split=='train' and args.blur: # if --blur=True in CLI arguments in training script, and only if we are in training
            if random.random() < args.blur_percent: #if random value between 0 and 1 is within our blur percent
                blurred_im_info = im_info.copy()
                blurred_im_info['img_id'] += '_blurred'
                blurred_im_info['filename'] = None  # No actual file, just augment in memory
                blurred_im_info['detections'] = detections  # No changes to bbox
                im_infos.append(blurred_im_info)
    print('Total {} images found'.format(len(im_infos)))
    return im_infos


class CitypersonsDataset(Dataset):
    def __init__(self, split, im_dir, ann_file):
        self.split = split
        self.im_dir = im_dir
        self.ann_file = ann_file
        classes = ['person']
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, ann_file)
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info['filename'])
        if im_info['filename'] is not None: # if it is NOT augmented data
            im = Image.open(im_info['filename'])
        elif im_info['img_id'][-8:] == '_flipped': # else if it is a flipped image:
            original_im_info = next(item for item in self.images_info if item['img_id'] == im_info['img_id'].replace('_flipped', ''))
            im = Image.open(original_im_info['filename']).transpose(Image.FLIP_LEFT_RIGHT) # flip the image
        elif im_info['img_id'][-8:] == '_blurred': # else if it is a blurred image:
            original_im_info = next(item for item in self.images_info if item['img_id'] == im_info['img_id'].replace('_blurred', ''))
            im = Image.open(original_im_info['filename']).filter(ImageFilter.GaussianBlur(radius=2)) # blur the image with a radius of 2

        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])

        return im_tensor, targets, im_info['filename']
        
