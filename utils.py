import torch.utils.data as data
import numpy as np
from PIL import Image
import os


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
