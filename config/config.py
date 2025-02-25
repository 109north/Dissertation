import argparse

parser = argparse.ArgumentParser()
# arguments for train_torchvision_frcnn.py
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
# arguments for infer_torchvision_frcnn.py
parser.add_argument('--config', dest='config_path',
                    default='config/citypersons.yaml', type=str)
parser.add_argument('--evaluate', dest='evaluate',
                    default=False, type=bool)
parser.add_argument('--infer_samples', dest='infer_samples',
                    default=True, type=bool)
parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                    required=True, type=str,
                    help='Path to the custom pretrained model checkpoint')
args = parser.parse_args()
