import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
from models import NormalisedVGG, Decoder
from utils import Dataset, InfiniteSampler
from ops import style_swap, TVloss, learning_rate_decay
import argparse
from tqdm import tqdm
import os
from PIL import Image


parser = argparse.ArgumentParser(description='Style Swap')
parser.add_argument('--content-dir', type=str, required=True, help='Content images for training')
parser.add_argument('--style-dir', type=str, required=True, help='Style images for training')
parser.add_argument('--content-test-dir', type=str, default='./inputs/content', help='Content test images for training')
parser.add_argument('--style-test-dir', type=str, default='./inputs/style', help='Style test images for training')
parser.add_argument('--encoder-path', type=str, default='./encoder/vgg_normalised_conv5_1.pth')
parser.add_argument('--max-iter', type=int, default=80000)
parser.add_argument('--image-size', type=int, default=512)
parser.add_argument('--crop-size', type=int, default=256)
parser.add_argument('--target-layer', type=str, default='relu3_1', help='Target hidden layer')
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--gpu', type=str, default=0)
parser.add_argument('--nThreads', type=int, default=4)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--learning-rate-decay', type=float, default=1e-4)
parser.add_argument('--tv', type=float, default=1e-6)
parser.add_argument('--pixel-loss', type=float, default=0)
parser.add_argument('--model-save-iter', type=int, default=1000)
parser.add_argument('--model-save-dir', type=str, default='./decoder')
parser.add_argument('--test-iter', type=int, default=1000)
parser.add_argument('--test-save-dir', type=str, default='./test')
parser.add_argument('--print-iter', type=int, default=500)

args = parser.parse_args()

device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.test_save_dir):
    os.mkdir(args.test_save_dir)

transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.RandomCrop(args.crop_size),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.ToTensor()
])

content_dataset = Dataset(args.content_dir, transform)
style_dataset = Dataset(args.style_dir, transform)

content_data_loader = iter(data.DataLoader(
    content_dataset,
    batch_size=args.batch_size,
    num_workers=args.nThreads,
    sampler=InfiniteSampler(content_dataset)
))

style_data_loader = iter(data.DataLoader(
    style_dataset,
    batch_size=args.batch_size,
    num_workers=args.nThreads,
    sampler=InfiniteSampler(style_dataset)
))

encoder = NormalisedVGG(pretrained_path=args.encoder_path).to(device)
decoder = Decoder().to(device)

optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)

criterion = nn.MSELoss()

for global_step in tqdm(range(args.max_iter)):

    c_batch = next(content_data_loader).to(device)
    s_batch = next(style_data_loader).to(device)

    c_latent = encoder(c_batch, args.target_layer)
    s_latent = encoder(s_batch, args.target_layer)

    ss = []
    for i in range(args.batch_size):
        for j in range(args.batch_size):
            c_latent_i = c_latent[i].unsqueeze(0)
            s_latent_j = c_latent[j].unsqueeze(0)
            ss.append(style_swap(c_latent_i, s_latent_j, 3))

    ss = torch.cat(ss, 0)

    reconstructed_ss = decoder(ss)
    reconstructed_ss_latent = encoder(reconstructed_ss, args.target_layer)

    loss = criterion(ss, reconstructed_ss_latent)

    if args.pixel_loss > 0:

        reconstructed_c = decoder(c_latent)
        reconstructed_s = decoder(s_latent)

        c_pixel_loss = criterion(c_batch, reconstructed_c)
        s_pixel_loss = criterion(s_batch, reconstructed_s)

        loss += (c_pixel_loss + s_pixel_loss) * args.pixel_loss

    if args.tv > 0:

        loss += TVloss(reconstructed_ss, args.tv)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lr = learning_rate_decay(optimizer, args.learning_rate, global_step, args.learning_rate_decay)

    if global_step % args.print_iter == 0:
        tqdm.write('step: %s, lr: %f, loss: %f' % (global_step, lr, loss.item()))

    if global_step % args.model_save_iter == 0:
        pass

    if global_step % args.test_iter == 0:
        c_test_data = os.listdir(args.content_test_dir)
        s_test_data = os.listdir(args.style_test_dir)
        for i in range(len(c_test_data)):
            c_test_img_path = os.path.join(args.content_test_dir, c_test_data[i])
            s_test_img_path = os.path.join(args.style_test_dir, s_test_data[i])
            c_test_img = test_transform(Image.open(c_test_img_path)).unsqueeze(0).to(device)
            s_test_img = test_transform(Image.open(s_test_img_path)).unsqueeze(0).to(device)

            c_test_latent = encoder(c_test_img)
            s_test_latent = encoder(s_test_img)
            ss = style_swap(c_test_latent, s_test_latent, 3)
            reconstructed_ss = decoder(ss).squeeze(0)
            transforms.ToPILImage()(reconstructed_ss).save('%s/%s_%s.jpg' % (args.test_save_dir, global_step, i))