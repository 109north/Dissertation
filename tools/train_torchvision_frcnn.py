#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import argparse
import os
import sys
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
import pandas as pd

# move into the root directory to find my data module
#os.chdir('/Users/narayanmurti/Workspace/Dissertation')
#sys.path.append(os.getcwd())
from config.config import args
from data.citypersons import CitypersonsDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    citypersons = CitypersonsDataset(split = 'train',
                     im_dir=dataset_config['im_train_path'],
                     ann_file=dataset_config['ann_train_path'])

    train_dataset = DataLoader(citypersons,
                               batch_size=8,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=collate_function)

    citypersons_test = CitypersonsDataset(split='test',
                                          im_dir=dataset_config['im_test_path'],
                                          ann_file=dataset_config['ann_test_path'])

    test_dataset = DataLoader(citypersons_test,
                              batch_size=1,
                              shuffle=False,
                              num_workers=4,
                              collate_fn=collate_function)


    if args.use_resnet50_fpn:
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                                 min_size=600,
                                                                                 max_size=1000,
        )
        faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
            faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=21)
    else:
        backbone = torchvision.models.resnet34(pretrained=True, norm_layer=torchvision.ops.FrozenBatchNorm2d)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        backbone.out_channels = 256
        roi_align = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        rpn_anchor_generator = AnchorGenerator()
        faster_rcnn_model = torchvision.models.detection.FasterRCNN(backbone,
                                                                    num_classes=21,
                                                                    min_size=600,
                                                                    max_size=1000,
                                                                    rpn_anchor_generator=rpn_anchor_generator,
                                                                    rpn_pre_nms_top_n_train=12000,
                                                                    rpn_pre_nms_top_n_test=6000,
                                                                    box_batch_size_per_image=128,
                                                                    rpn_post_nms_top_n_test=300
                                                                    )

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(lr=1E-4,
                                params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=5E-5, momentum=0.9)

    num_epochs = train_config['num_epochs']
    step_count = 0

    rpn_classification_epochs = []
    frcnn_classification_epochs = []
    test_rpn_classification_epochs = []
    test_frcnn_classification_epochs = []

    for i in range(num_epochs):
        #training phase
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

            loss.backward()
            optimizer.step()
            step_count +=1
        print('Finished epoch {}'.format(i))
        if args.use_resnet50_fpn:
            torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                    'tv_frcnn_r50fpn_' + train_config['ckpt_name']))
        else:
            torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                    'tv_frcnn_' + train_config['ckpt_name']))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)

        #Testing phase
        test_frcnn_classification_losses = []
        test_rpn_classification_losses = []

        with torch.no_grad():
            for ims, targets, _ in tqdm(test_dataset):
                for target in targets:
                    target['boxes'] = target['bboxes'].float().to(device)
                    del target['bboxes']
                    target['labels'] = target['labels'].long().to(device)
                images = [im.float().to(device) for im in ims]
                batch_losses = faster_rcnn_model(images, targets)

                test_frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
                test_rpn_classification_losses.append(batch_losses['loss_objectness'].item())  # Store RPN loss

        print(f"  Test - RPN Classification Loss: {np.mean(test_rpn_classification_losses):.4f} | "
              f"Test - FRCNN Classification Loss: {np.mean(test_frcnn_classification_losses):.4f}")
        
        rpn_classification_epochs.append(np.mean(rpn_classification_losses))
        frcnn_classification_epochs.append(np.mean(frcnn_classification_losses))
        test_rpn_classification_epochs.append(np.mean(test_rpn_classification_losses))
        test_frcnn_classification_epochs.append(np.mean(test_frcnn_classification_losses))
        
    print('Done Training...')

    losses_dict = {'train rpn': rpn_classification_epochs,
                   "train frcnn": frcnn_classification_epochs,
                   "test rpn": test_rpn_classification_epochs,
                   "test frcnn": test_frcnn_classification_epochs}
    losses_df = pd.DataFrame(losses_dict)
    losses_df.to_csv('/home/nam27/Dissertation/results/losses.csv')
    


#THIS IS FOR RUNNING ON THE COMMAND LINE
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path',
                        default='config/citypersons.yaml', type=str)
    parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                        default=True, type=bool)
    parser.add_argument('--flip', dest='flip',
                        default=False, type=bool, 
                        help='Run augmentation with flipping?')
    parser.add_argument('--flip_percent', dest='flip_percent',
                        default=0.25, type=float,
                        help='percent chance for a photo to be duplicated and flipped')
    parser.add_argument('--blur', dest='blur',
                        default=False, type=bool, 
                        help='Run augmentation with blurring?')
    parser.add_argument('--blur_percent', dest='blur_percent',
                        default=0.25, type=float,
                        help='percent chance for a photo to be duplicated and blurred')
    parser.add_argument('--brightness', dest='brightness',
                        default=False, type=bool,
                        help='Run augmentation with brightness?')
    parser.add_argument('--brightness_percent', dest='brightness_percent',
                        default=0.25, type=float,
                        help='percent change for a photo to be duplicated and brightened/darkened')
    parser.add_argument('--bright_low', dest='bright_low',
                        default=0.5, type=float,
                        help='lower bound for brightness amount (with 1 being no change)')
    parser.add_argument('--bright_high', dest='bright_high',
                        default=1.5, type=float,
                        help='upper bound for brightness amount (with 1 being no change)')
    parser.add_argument('--augmix', dest='augmix',
                        default=False, type=bool,
                        help='Run augmentation with AugMix?')
    parser.add_argument('--augmix_percent', dest='augmix_percent',
                        default=0.25, type=float,
                        help='percent chance for a photo to be duplicated and AugMixed')
    parser.add_argument('--shrink_bbox', dest='shrink_bbox',
                        default=False, type=bool,
                        help='Run augmentation with shrinking?')
    parser.add_argument('--shrink_bbox_percent', dest='shrink_bbox_percent',
                        default=0.25, type=float,
                        help='percent chance for a photo to be duplicated and shrunken')
    args = parser.parse_args(args=[] if sys.argv[0].endswith('ipykernel_launcher.py') else sys.argv[1:])
    train(args)


#THIS IS FOR RUNNING IN JUPYTER
# Manually define the arguments instead of using argparse
#class Args:
#    config_path = 'config/citypersons.yaml'
#    use_resnet50_fpn = True  # or False

#args = Args()
#train(args)






