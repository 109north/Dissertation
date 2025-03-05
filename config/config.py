import argparse

parser = argparse.ArgumentParser()
# arguments used by both documents
parser.add_argument('--config', dest='config_path',
                    default='config/citypersons.yaml', type=str)
parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                    default=True, type=bool)

# arguments for train_torchvision_frcnn.py
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
parser.add_argument('--augmix', dest='augmix',
                    default=False, type=bool,
                    help='Run augmentation with AugMix?')
parser.add_argument('--augmix_percent', dest='augmix_percent',
                    default=0.25, type=float,
                    help='percent chance for a photo to be duplicated and AugMixed')

# arguments for infer_torchvision_frcnn.py
parser.add_argument('--evaluate', dest='evaluate',
                    default=False, type=bool)
parser.add_argument('--infer_samples', dest='infer_samples',
                    default=True, type=bool)
parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                    type=str,
                    help='Path to the custom pretrained model checkpoint')

# arguments for extract_image
parser.add_argument('--select_image_filename', dest='select_image_filename',
                    default=True, type=str,
                   help="ensure you enter the entire filepath (relative to the GPU")

args = parser.parse_args()
