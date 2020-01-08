import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
from models import NormalisedVGG, Decoder
from utils import forward_transform
from ops import style_swap
import argparse

parser = argparse.ArgumentParser(description='Style Swap')
parser.add_argument('--model-path', type=str, default='./decoder/decoder.5375', help='path to decoder')
parser.add_argument('--content-path', type=str, required=True, help='path to content image')
parser.add_argument('--style-path', type=str, required=True, help='path to style image')
parser.add_argument('--save-path', type=str)
parser.add_argument('--image-size', type=int, default=512)
parser.add_argument('--patch-size', type=int, default=3)
parser.add_argument('--stride', type=int, default=3)
parser.add_argument('--gpu', type=str, default=0)

args = parser.parse_args()

device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() else 'cpu')

encoder = NormalisedVGG().to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))

forward_transform(encoder, decoder, args.content_path, args.style_path, args.image_size,
                  device, style_swap, args.save_path, args.patch_size, args.stride)
