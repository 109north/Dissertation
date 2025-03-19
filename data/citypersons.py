#!/usr/bin/env python
# coding: utf-8

# In[149]:


import glob
import os
import random
import pandas as pd
import cv2
import warnings
import numpy as np
from config.config import args

import torch
import torchvision
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

from torchvision import transforms

# code for person shrinking augmentation
def shrink_bboxes_in_image(im, detections, scale_range=(0.1, 0.4), shrink_prob=0.75):
    """
    Randomly selects bounding boxes, shrinks them down, and pastes them back within their original bounding box
    while keeping the rest of the image unchanged.
    
    :param im: PIL Image
    :param detections: List of bounding box dictionaries
    :param scale_range: Tuple (min_scale, max_scale) for shrinking bbox
    :param shrink_prob: Probability of shrinking each bounding box
    :return: Image with shrunken bounding boxes and updated detections
    """
    w, h = im.size  # Original image size
    im_array = np.array(im)  # Convert the original image to an array
        
    new_detections = []

    for det in detections:
        if random.random() < shrink_prob:  # Apply shrinking with a given probability
            x1, y1, x2, y2 = det['bbox']
            # if the bounding box goes off the screen left or right, clip it into the image dimensions
            if x1 < 0:
                x1 = 1
            if x2 > 2048:
                x2 = 2048
            person_crop = im_array[y1:y2, x1:x2].copy()  # Extract the person region

            if person_crop.size == 0 or (y2 - y1) <= 0 or (x2 - x1) <= 0:
                print(f"Skipping empty bbox: {x1, y1, x2, y2}")
                new_detections.append(det)
                continue

            
            # Choose a random shrinking scale
            scale = random.uniform(*scale_range)
            new_w, new_h = max(1, int((x2 - x1) * scale)), max(1, int((y2 - y1) * scale))

            # Ensure new width/height are within original bbox
            new_w, new_h = max(1, min(new_w, x2 - x1)), max(1, min(new_h, y2 - y1))
            
            # Pick a random position INSIDE the original bounding box
            new_x1 = x1 + random.randint(0, (x2 - x1) - new_w)
            new_y1 = y1 + random.randint(0, (y2 - y1) - new_h)
            new_x2, new_y2 = new_x1 + new_w, new_y1 + new_h
            
            # Black out the area inside the original bounding box
            im_array[y1:y2, x1:x2] = 0  # Set the original area to black
            
            # Resize the cropped person
            person_crop_resized = cv2.resize(person_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Fix size mismatch issue before pasting
            if person_crop_resized.shape[:2] != (new_y2 - new_y1, new_x2 - new_x1):
                print(f"Mismatch detected! Resizing again: Expected {(new_y2 - new_y1, new_x2 - new_x1)}, Got {person_crop_resized.shape}")
                person_crop_resized = cv2.resize(person_crop_resized, (new_x2 - new_x1, new_y2 - new_y1), interpolation=cv2.INTER_AREA)

            # Paste the resized person back into the blacked-out region
            try:
                im_array[new_y1:new_y2, new_x1:new_x2] = person_crop_resized # Paste tiny person
            except ValueError as e:
                print(f"Error pasting resized person: {e}")
                print(f"Expected shape: {new_y2-new_y1, new_x2-new_x1}, Got: {person_crop_resized.shape}")
            
            # Update the bounding box to match the new size and position
            new_detections.append({'label': det['label'], 'bbox': [new_x1, new_y1, new_x2, new_y2]})
        else:
            # If no shrinking, keep the original bounding box (no changes)
            new_detections.append(det)

    # Convert the NumPy array back to PIL Image
    shrunken_im = Image.fromarray(im_array)
    
    return shrunken_im, new_detections
    

def load_images_and_anns(im_dir, ann_file, split):
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
        #h, w = im_tensor.shape[:2] # get height and width of image
        w, h = im.size
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
        
        if split=='train' and args.flip: #if --flip=True in CLI arguments in training script, and only if we are in training
            if random.random() < args.flip_percent: #if random value between 0 and 1 is within our flip percent
                flipped_im_info = im_info.copy() 
                flipped_im_info['img_id'] += '_flipped' #change the image id
                flipped_im_info['filename'] = None  # No actual file, just augment in memory
                flipped_detections = []

                for det in detections:
                    # flip the bbox
                    x1, y1, x2, y2 = det['bbox']
                    flipped_x1 = w - x2
                    flipped_x2 = w - x1
                    #bbox_width = x2 - x1 
                    #flipped_x1 = w - x2
                    #flipped_x2 = flipped_x1 + bbox_width
                    
                    flipped_detections.append({'label': det['label'], 'bbox': [flipped_x1, y1, flipped_x2, y2]})
            
                flipped_im_info['detections'] = flipped_detections
                im_infos.append(flipped_im_info) # append the flipped info into all the image infos
        
        if split=='train' and args.blur: # if --blur=True in CLI arguments in training script, and only if we are in training
            if random.random() < args.blur_percent: #if random value between 0 and 1 is within our blur percent
                blurred_im_info = im_info.copy()
                blurred_im_info['img_id'] += '_blurred'
                blurred_im_info['filename'] = None  # No actual file, just augment in memory
                blurred_im_info['detections'] = detections  # No changes to bbox
                im_infos.append(blurred_im_info)

        if split=='train' and args.brightness: # if --brightness=True in CLI arguments and only if we are in training
            if random.random() < args.brightness_percent:
                brightness_im_info = im_info.copy()
                brightness_factor = random.uniform(args.bright_low, args.bright_high)  # Randomly choose between darkening (bright_low) and brightening (bright_high)
                
                if brightness_factor > 1.0:
                    brightness_im_info['img_id'] += '_brightened'
                else:
                    brightness_im_info['img_id'] += '_darkened'

                brightness_im_info['filename'] = None  # No actual file, just augment in memory
                brightness_im_info['detections'] = detections  # No changes to bbox
                brightness_im_info['brightness_factor'] = brightness_factor  # Store brightness factor
                im_infos.append(brightness_im_info)

        if split=='train' and args.augmix: #if --augmix=True in CLI and only if we are in training
            if random.random() < args.augmix_percent:
                augmix_im_info = im_info.copy()
                augmix_im_info['img_id'] += '_augmix'  # Tag the image
                augmix_im_info['filename'] = None  # No actual file, augment in memory
                augmix_im_info['detections'] = detections  # No changes to bounding boxes
                im_infos.append(augmix_im_info)  # Add AugMix-augmented image alongside original

        if split == 'train' and args.shrink_bbox:  # If shrinking augmentation is enabled
            if random.random() < args.shrink_bbox_percent:  # Apply with a certain probability
                shrunken_im_info = im_info.copy()
                shrunken_im_info['img_id'] += '_shrunken'
                shrunken_im, shrunken_detections = shrink_bboxes_in_image(im, detections)

                if shrunken_im is not None and shrunken_detections:
                    shrunken_im_info['filename'] = None  # No actual file, augment in memory
                    shrunken_im_info['detections'] = shrunken_detections  # Store adjusted bounding boxes
                    im_infos.append(shrunken_im_info)  # Append augmented image


            
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
        self.images_info = load_images_and_anns(im_dir, ann_file, self.split)
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        im_info = self.images_info[index]
        if im_info['filename'] is not None: # if it is NOT augmented data
            im = Image.open(im_info['filename'])
        elif im_info['img_id'][-8:] == '_flipped': # else if it is a flipped image:
            original_im_info = next(item for item in self.images_info if item['img_id'] == im_info['img_id'].replace('_flipped', ''))
            im = Image.open(original_im_info['filename']).transpose(Image.FLIP_LEFT_RIGHT) # flip the image
        elif im_info['img_id'][-8:] == '_blurred': # else if it is a blurred image:
            original_im_info = next(item for item in self.images_info if item['img_id'] == im_info['img_id'].replace('_blurred', ''))
            im = Image.open(original_im_info['filename']).filter(ImageFilter.GaussianBlur(radius=4)) # blur the image with a radius of 2
        elif im_info['img_id'][-11:] == '_brightened' or im_info['img_id'][-9:] == '_darkened': # else if its brightness aug
            original_im_info = next(item for item in self.images_info if item['img_id'] == im_info['img_id'].replace('_brightened', '').replace('_darkened', ''))
            im = Image.open(original_im_info['filename'])
            enhancer = ImageEnhance.Brightness(im)
            im = enhancer.enhance(im_info['brightness_factor'])  # Apply brightness factor
        elif im_info['img_id'][-7:] == '_augmix': # else if it is an augmix image
            original_im_info = next(item for item in self.images_info if item['img_id'] == im_info['img_id'].replace('_augmix', ''))
            im = Image.open(original_im_info['filename'])
            augmix_transform = transforms.AugMix()  # Create the transform object
            im = augmix_transform(im)  # Apply AugMix to the image
        elif im_info['img_id'][-9:] == '_shrunken':  # else if it is a shrunken image
            original_im_info = next(item for item in self.images_info if item['img_id'] == im_info['img_id'].replace('_shrunken', ''))
            im = Image.open(original_im_info['filename'])
            # Apply the shrinking augmentation (shrinking bounding box)
            im, new_detections = shrink_bboxes_in_image(im, original_im_info['detections'])
            im_info['detections'] = new_detections  # Update the detections for the shrunken image


            

        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])

        return im_tensor, targets, im_info['filename']
        
