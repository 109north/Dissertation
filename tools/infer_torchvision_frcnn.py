#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import cv2
import torchvision
import argparse
import random
import os
import yaml
import pandas
from tqdm import tqdm
from model.faster_rcnn import FasterRCNN
from data.citypersons import CitypersonsDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, score, im_name], ...],
    #       'car' : [[x1, y1, x2, y2, score, im_name], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, im_name], ...],
    #       'car' : [[x1, y1, x2, y2, im_name], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    # average precisions for ALL classes
    aps = []
    det_df_list = []
    fn_df_list = [] # List to store false negatives
    all_recalls = {}
    all_precisions = {}
    
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0, im_name]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M, im_name]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0, im_name]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N, im_name]),
        #   ...
        # ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-2])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-2], gt_box[:-1])
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True

            # append det_idx, image name, bbox, score, max iou found, tp, fp into det_df
            det_df_list.append([det_idx, det_pred[5], det_pred[0], det_pred[1], det_pred[2],
                           det_pred[3], det_pred[4], max_iou_found, tp[det_idx], 
                           fp[det_idx]])
            
        # Identify false negatives (GT boxes never matched)
        for im_idx, im_gts in enumerate(gt_boxes):
            for gt_idx, (gt_box, matched) in enumerate(zip(im_gts[label], gt_matched[im_idx])):
                if not matched:
                    fn_df_list.append([gt_box[4], gt_box[0], gt_box[1], gt_box[2], gt_box[3]])
            
        # Convert det_df to pandas dataframe and save as csv to GPU path
        det_df = pandas.DataFrame(det_df_list)
        det_df = det_df.rename(columns={0:'det_idx', 1:'filename', 2:'x1', 3:'y1', 4:'x2', 5:'y2', 
                                        6:'confidence score', 7:'max_iou', 8:'tp', 9:'fp'})
        det_df.to_csv('/home/nam27/Dissertation/results/det_df.csv')

        fn_df = pandas.DataFrame(fn_df_list, columns=['filename', 'x1', 'y1', 'x2', 'y2'])
        fn_df.to_csv('/home/nam27/Dissertation/results/fn_df.csv')
        
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        all_recalls[label] = recalls[-1] if len(recalls) > 0 else 0.0
        all_precisions[label] = precisions[-1] if len(precisions) > 0 else 0.0

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps, all_recalls, all_precisions


def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    citypersons = CitypersonsDataset('test', im_dir=dataset_config['im_test_path'], ann_file=dataset_config['ann_test_path'])
    test_dataset = DataLoader(citypersons, batch_size=1, shuffle=False)

    if args.use_resnet50_fpn:
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                                 min_size=600,
                                                                                 max_size=1000,
                                                                                 box_score_thresh=0.7,
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
                                                                    box_roi_pool=roi_align,
                                                                    rpn_pre_nms_top_n_train=12000,
                                                                    rpn_pre_nms_top_n_test=6000,
                                                                    box_batch_size_per_image=128,
                                                                    box_score_thresh=0.7,
                                                                    rpn_post_nms_top_n_test=300)

    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    #THIS IS WHERE I SPECIFY WHICH CUSTOM PRETRAINED MODEL I WILL USE TO TEST
    # Load the model from the specified checkpoint path
    faster_rcnn_model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    #if args.use_resnet50_fpn:
    #    faster_rcnn_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                                              'tv_frcnn_r50fpn_' + train_config['ckpt_name']),
    #                                                 map_location=device))
    #else:
    #    faster_rcnn_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                                              'tv_frcnn_' + train_config['ckpt_name']),
    #                                                 map_location=device))
    return faster_rcnn_model, citypersons, test_dataset


def infer(args):
    if args.use_resnet50_fpn:
        output_dir = 'samples_tv_r50fpn'
    else:
        output_dir = 'samples_tv'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    faster_rcnn_model, citypersons, test_dataset = load_model_and_dataset(args)

    
    
    #extract the least confidence prediction boxes. the worst 6
    det_df_path = '/home/nam27/Dissertation/results/det_df.csv'
    det_df = pandas.read_csv(det_df_path)
    last_6_filenames = list(det_df.tail(6)['filename'])
    sample_count = 0

    for fname in last_6_filenames:
        fname = fname.translate({ord(i): None for i in "(),'"}) #remove the extra characters from the filename
        matching_index = [i for i, info in enumerate(citypersons.images_info) if info['filename'] == fname] #find the index of the corresponding filename
        index = matching_index[0]
        im, target, _ = citypersons[index]
        im = im.unsqueeze(0).float().to(device)
        
        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()

        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = citypersons.idx2label[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=citypersons.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('{}/worst_six_output_gt_{}.png'.format(output_dir, sample_count), gt_im)

        # Getting predictions from trained model
        frcnn_output = faster_rcnn_model(im, None)[0]
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        im = cv2.imread(fname)
        im_copy = im.copy()

        # Saving images with predicted boxes
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(citypersons.idx2label[labels[idx].detach().cpu().item()],
                                        scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite('{}/worst_six_output_{}.jpg'.format(output_dir, sample_count), im)

        sample_count += 1


    #extract the BEST TWO results
    best_2_filenames = list(det_df.head(2)['filename'])
    sample_count = 0
    
    for fname in best_2_filenames:
        fname = fname.translate({ord(i): None for i in "(),'"}) #remove the extra characters from the filename
        matching_index = [i for i, info in enumerate(citypersons.images_info) if info['filename'] == fname] #find the index of the corresponding filename
        index = matching_index[0]
        im, target, _ = citypersons[index]
        im = im.unsqueeze(0).float().to(device)
        
        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()

        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = citypersons.idx2label[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=citypersons.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('{}/best_two_output_gt_{}.png'.format(output_dir, sample_count), gt_im)

        # Getting predictions from trained model
        frcnn_output = faster_rcnn_model(im, None)[0]
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        im = cv2.imread(fname)
        im_copy = im.copy()

        # Saving images with predicted boxes
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(citypersons.idx2label[labels[idx].detach().cpu().item()],
                                        scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite('{}/best_two_output_{}.jpg'.format(output_dir, sample_count), im)

        sample_count += 1





def evaluate_map(args):
    faster_rcnn_model, citypersons, test_dataset = load_model_and_dataset(args)
    gts = []
    preds = []
    for im, target, fname in tqdm(test_dataset):
        im_name = fname
        im = im.float().to(device)
        target_boxes = target['bboxes'].float().to(device)[0]
        target_labels = target['labels'].long().to(device)[0]
        frcnn_output = faster_rcnn_model(im, None)[0]

        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']

        pred_boxes = {}
        gt_boxes = {}
        for label_name in citypersons.label2idx:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            label_name = citypersons.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, score, im_name])
        for idx, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = citypersons.idx2label[label]
            gt_boxes[label_name].append([x1, y1, x2, y2, im_name])

        gts.append(gt_boxes)
        preds.append(pred_boxes)

    mean_ap, all_aps, all_recalls, all_precisions = compute_map(preds, gts, method='interp')
    print('Class Wise Average Precisions')
    for idx in range(len(citypersons.idx2label)):
        print('AP for class {} = {:.4f}'.format(citypersons.idx2label[idx], all_aps[citypersons.idx2label[idx]]))
        print('Recall for class {} = {:.4f}'.format(citypersons.idx2label[idx], all_recalls[citypersons.idx2label[idx]]))
        print('Precision for class {} = {:.4f}'.format(citypersons.idx2label[idx], all_precisions[citypersons.idx2label[idx]]))
    print('Mean Average Precision : {:.4f}'.format(mean_ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inference using torchvision code faster rcnn')
    parser.add_argument('--config', dest='config_path',
                        default='config/citypersons.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=False, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=bool)
    parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                        default=True, type=bool)
    # add a required argument to specify the path of the pretrained model checkpoint to test on
    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                    required=True, type=str,
                    help='Path to the custom pretrained model checkpoint')
    args = parser.parse_args()

    if args.evaluate:
        evaluate_map(args)
    else:
        print('Not Evaluating as `evaluate` argument is False')
    
    if args.infer_samples:
        infer(args)
    else:
        print('Not Inferring for samples as `infer_samples` argument is False')

