import argparse

parser = argparse.ArgumentParser()
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
args = parser.parse_args()
