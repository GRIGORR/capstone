import numpy as np
import json
from copy import deepcopy
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


def get_datasets(root, dataset='cifar10', cutout=-1):
    mean_map = {'cifar10': [x / 255 for x in [125.3, 123.0, 113.9]],
                'cifar100': [x / 255 for x in [129.3, 124.1, 112.4]]}
    std_map = {'cifar10': [x / 255 for x in [63.0, 62.1, 66.7]],
                'cifar100': [x / 255 for x in [68.2, 65.4, 70.4]]}

    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean_map[dataset], std_map[dataset])]
    if cutout > 0:
        lists += [CUTOUT(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean_map[dataset], std_map[dataset])])

    if dataset == 'cifar10':
        train_data = dset.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif dataset == 'cifar100':
        train_data = dset.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(root, train=False, transform=test_transform, download=True)

    else:
        raise ValueError('Invalid dataset. Should be either cifar10 or cifar100')
    assert len(train_data) == 50000 and len(test_data) == 10000
    return train_data, test_data, (1, 3, 32, 32), int(dataset.split('cifar')[1])


def get_nas_search_loaders(train_data, valid_data, dataset, config_root, batch_size, workers):
    with open(config_root, 'r') as f:
        cifar_split = json.load(f)
    if dataset == 'cifar10':
        train_split, valid_split = cifar_split['train'], cifar_split['valid']
        search_data = SearchDataset(train_data, train_split, valid_split)
        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch_size, shuffle=True,
                                                    num_workers=workers, pin_memory=True)
    elif dataset == 'cifar100':
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data)
        search_valid_data.transform = train_data.transform
        search_data = SearchDataset(dataset, [search_train_data, search_valid_data],
                                    list(range(len(search_train_data))), cifar_split.valid_split)

        search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch_size, shuffle=True,
                                                    num_workers=workers, pin_memory=True)
    return search_loader


class SearchDataset(data.Dataset):

    def __init__(self, data, train_split, valid_split):
        self.data = data
        self.train_split = train_split
        self.valid_split = valid_split
        intersection = set(train_split).intersection(set(valid_split))
        assert len(intersection) == 0, 'the splitted train and validation sets should have no intersection'
        self.length = len(self.train_split)

    def __repr__(self):
        return ('{name}(name={datasetname}, train={tr_L}, valid={val_L}, version={ver})'.format(
            name=self.__class__.__name__, dtr_L=len(self.train_split),
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
