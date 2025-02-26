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



# RUN FILE WITH --checkpoint_path argument for where to get the model from.
# and --select_image_filename the filename of the iamge



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









def get_image(args):
  output_dir = 'extract_images'
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  faster_rcnn_model, citypersons, test_dataset = load_model_and_dataset(args)

  fname = args.select_image_filename

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
  cv2.imwrite('{}/extract_gt_{}.png'.format(output_dir, fname), gt_im)

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
  cv2.imwrite('{}/extract_{}.jpg'.format(output_dir, fname), im)









if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', dest='config_path',
                      default='config/citypersons.yaml', type=str)
  parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                      default=True, type=bool)
  parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                      type=str,
                      help='Path to the custom pretrained model checkpoint')
  parser.add_argument('--select_image_filename', dest='select_image_filename',
                      default=True, type=str)
  args = parser.parse_args()

  get_image(args)
