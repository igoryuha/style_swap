import torch
from torchvision import transforms
from models import NormalisedVGG, Decoder
from utils import Dataset
import argparse

parser = argparse.ArgumentParser(description='Style Swap')
parser.add_argument('--content-dir', type=str, required=True, help='Content images for training')
parser.add_argument('--style-dir', type=str, required=True, help='Style images for training')
parser.add_argument('--content-test-dir', type=str, help='Content test images for training')
parser.add_argument('--style-test-dir', type=str, help='Style test images for training')
parser.add_argument('--max-iter', type=int, default=80000)
parser.add_argument('--image-size', type=int, default=512)
parser.add_argument('--crop-size', type=int, default=256)
parser.add_argument('--target-layer', type=str, default='relu3_1', help='Target hidden layer')
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--gpu', type=str, default=0)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--learning-rate-decay', type=float, default=1e-4)
parser.add_argument('--tv', type=float, default=1e-6)
parser.add_argument('--pixel-loss', type=float, default=0)
parser.add_argument('--save-iter', type=int, default=1000)
parser.add_argument('--print-iter', type=int, default=500)

args = parser.parse_args()

device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.RandomCrop(args.crop_size),
    transforms.ToTensor()
])

content_dataset = Dataset(args.content_dir, transform)
style_dataset = Dataset(args.style_dir, transform)

encoder = NormalisedVGG()
decoder = Decoder()
