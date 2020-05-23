import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import json
from SearchDatasetWrap import SearchDataset
from config_utils import load_config


class CUTOUT(object):

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return '{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__)

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_datasets(root, cutout):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean, std)]
    if cutout > 0:
        lists += [CUTOUT(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = dset.CIFAR10(root, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(root, train=False, transform=test_transform, download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000

    return train_data, test_data, (1, 3, 32, 32), 10


def get_nas_search_loaders(train_data, config_root, batch_size, workers):

    # split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
    # cifar_split = load_config(config_root, None, None)
    with open(config_root, 'r') as f:
        cifar_split = json.load(f)
    train_split, valid_split = cifar_split['train'], cifar_split['valid']  # search over the proposed training and validation set
    # logger.log('Load split file from {:}'.format(split_Fpath))      # they are two disjoint groups in the original CIFAR-10 training set
    search_data = SearchDataset(train_data, train_split, valid_split)
    # data loader
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                                                pin_memory=True)
    return search_loader


# if __name__ == '__main__':
#  train_data, test_data, xshape, class_num = dataset = get_datasets('cifar10', '/data02/dongxuanyi/.torch/cifar.python/', -1)
#  import pdb; pdb.set_trace()
