from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as tr
import pandas as pd
from PIL import Image
import os.path as osp
import medmnist

class ClassDataset(Dataset):
    def __init__(self, csv_path, data_path, transforms, tg_size):
        # assumes in the csv first column is file name, second column is target
        self.csv_path=csv_path
        df = pd.read_csv(self.csv_path)
        self.data_path = data_path
        self.im_list = df['image_id'].values
        self.targets = df['label'].values
        self.classes = list(df['label'].unique())
        self.transforms = transforms
        self.resize = tr.Resize(tg_size)
        self.tensorize = tr.ToTensor()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.normalize = tr.Normalize(mean, std)

    def __getitem__(self, index):
        # load image and targets
        img = Image.open(osp.join(self.data_path, self.im_list[index]))
        if 'gbcu' in self.data_path:
            img = img.convert('RGB')


        target = self.targets[index]

        img = self.resize(img)
        if self.transforms is not None:
            img = self.transforms(img)
        img = self.tensorize(img)
        img = self.normalize(img)
        return img, target

    def __len__(self):
        return len(self.im_list)

def get_class_loaders(csv_train, csv_val, data_path, tg_size, batch_size, num_workers, see_classes=True):
    # First dataset has TrivialAugment transforms, second loader has nothing (resize, tensor, normalize are inside)
    train_transforms = tr.TrivialAugmentWide()
    val_transforms = None

    train_dataset = ClassDataset(csv_train, data_path, train_transforms, tg_size)
    val_dataset = ClassDataset(csv_val, data_path, val_transforms, tg_size)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    if see_classes:
        print(20 * '*')
        for c in range(len(np.unique(train_dataset.targets))):
            exs_train = np.count_nonzero(train_dataset.targets == c)
            exs_val = np.count_nonzero(val_dataset.targets == c)
            print('Found {:d}/{:d} train/val examples of class {}'.format(exs_train, exs_val, c))

    return train_loader, val_loader

def get_class_test_loader(csv_test, data_path, tg_size, batch_size, num_workers):
    # resize, tensor, normalize are inside ClassDataset already
    test_transforms = None
    test_dataset = ClassDataset(csv_test, data_path, test_transforms, tg_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return test_loader

def get_medmnist_loaders(mnist_subdataset, batch_size, num_workers, see_classes=True, tg_size=28):

    info = medmnist.INFO[mnist_subdataset]
    classes = info['label']
    # required transforms
    tensorizer = tr.ToTensor()
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # not really necessary, since we do not use pretraining
    normalizer = tr.Normalize(mean, std)

    train_transforms = tr.Compose([tr.TrivialAugmentWide(), tensorizer, normalizer])
    val_transforms = tr.Compose([tensorizer, normalizer])
    # this is a quick fix because convnext needs 2^5 apparently
    if tg_size != 28:
        resizer = tr.Resize(tg_size)
        train_transforms = tr.Compose([tr.TrivialAugmentWide(), resizer, tensorizer, normalizer])
        val_transforms = tr.Compose([resizer, tensorizer, normalizer])

    DataClass = getattr(medmnist, info['python_class'])
    # load the data train_dataset = DataClass(split='train', transform=data_transform, as_rgb=True, download=download)
    train_dataset = DataClass(split='train', transform=train_transforms, as_rgb=True, download=True)
    val_dataset = DataClass(split='val', transform=val_transforms, as_rgb=True, download=True)

    # encapsulate data into dataloader form
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    setattr(train_dataset, 'classes', classes)
    setattr(val_dataset, 'classes', classes)

    if see_classes:
            print(20 * '*')
            for c in range(len(classes)):
                exs_train = np.count_nonzero(train_dataset.labels == c)
                exs_val = np.count_nonzero(val_dataset.labels == c)
                print('Found {:d}/{:d} train/val examples of class {}'.format(exs_train, exs_val, c))

    return train_loader, val_loader

def get_medmnist_test_loader(mnist_subdataset, batch_size, num_workers, tg_size=28):
    info = medmnist.INFO[mnist_subdataset]
    classes = info['label']

    # required transforms
    tensorizer = tr.ToTensor()
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    normalizer = tr.Normalize(mean, std)

    test_transforms = tr.Compose([tensorizer, normalizer])
    if tg_size != 28: # this is a quick fix because convnext needs 2^5 apparently
        resizer = tr.Resize(tg_size)
        test_transforms = tr.Compose([resizer, tensorizer, normalizer])
    
    DataClass = getattr(medmnist, info['python_class'])
    # load the data train_dataset = DataClass(split='train', transform=data_transform, as_rgb=True, download=download)

    test_dataset = DataClass(split='test', transform=test_transforms, as_rgb=True, download=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    setattr(test_dataset, 'classes', classes)

    return test_loader
