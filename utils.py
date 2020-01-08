import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import os


def eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])


class Dataset(data.Dataset):

    def __init__(self, root, transform=None):
        super(Dataset, self).__init__()
        self.data = os.listdir(root)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        full_path = os.path.join(self.root, self.data[index])
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def infinite_sampler(n):
    i = 0
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            order = np.random.permutation(n)
            i = 0


class InfiniteSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        super(InfiniteSampler, self).__init__(data_source)
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(infinite_sampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def deprocess(tensor):
    img = tensor.squeeze(0)
    img = torch.clamp(img, 0, 1).cpu()
    return transforms.ToPILImage()(img)


def forward_group_transform(encoder, decoder, content_dir, style_dir, image_size,
                      device, style_swap, save_dir, k_size=3, stride=3, global_step=None):

    content = sorted(os.listdir(content_dir))
    style = sorted(os.listdir(style_dir))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, (c, s) in enumerate(zip(content, style)):
        content_path = os.path.join(content_dir, c)
        style_path = os.path.join(style_dir, c)

        if global_step is not None:
            save_path = '%s/step_%s_%s_kSize%s_Stride%s.jpg' % (save_dir, global_step, i, k_size, stride)
        else:
            save_path = '%s/%s_kSize%s_Stride%s.jpg' % (save_dir, i, k_size, stride)

        forward_transform(encoder, decoder, content_path, style_path, image_size,
                          device, style_swap, save_path, k_size, stride)


def forward_transform(encoder, decoder, content_path, style_path, image_size,
                      device, style_swap, save_path, k_size, stride):

    transform = eval_transform(image_size)

    content = transform(Image.open(content_path))
    style = transform(Image.open(style_path))

    content = content.unsqueeze(0).to(device)
    style = style.unsqueeze(0).to(device)

    c_latent = encoder(content)
    s_latent = encoder(style)
    ss = style_swap(c_latent, s_latent, k_size, stride)
    reconstructed = decoder(ss)

    reconstructed = deprocess(reconstructed)

    reconstructed.save(save_path)
