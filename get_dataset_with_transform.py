import numpy as np
import json
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data


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
    with open(config_root, 'r') as f:
        cifar_split = json.load(f)
    train_split, valid_split = cifar_split['train'], cifar_split['valid']
    search_data = SearchDataset(train_data, train_split, valid_split)
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                                                pin_memory=True)
    return search_loader


class SearchDataset(data.Dataset):

    def __init__(self, data, train_split, valid_split):
        self.datasetname = 'cifar10'
        self.data = data
        self.train_split = train_split
        self.valid_split = valid_split
        intersection = set(train_split).intersection(set(valid_split))
        assert len(intersection) == 0, 'the splitted train and validation sets should have no intersection'
        self.length = len(self.train_split)

    def __repr__(self):
        return ('{name}(name={datasetname}, train={tr_L}, valid={val_L}, version={ver})'.format(
            name=self.__class__.__name__, datasetname=self.datasetname, tr_L=len(self.train_split),
            val_L=len(self.valid_split), ver=self.mode_str))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index >= 0 and index < self.length, 'invalid index = {:}'.format(index)
        train_index = self.train_split[index]
        valid_index = np.random.choice(self.valid_split)
        train_image, train_label = self.data[train_index]
        valid_image, valid_label = self.data[valid_index]

        return train_image, train_label, valid_image, valid_label
