import os
import csv
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

import params


class NIH(data.Dataset):

    def __init__(self, root, train=True, val=False, transform=None):
        """Init NIH dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.train = train
        self.val = val
        self.transform = transform
        self.dataset_size = None


        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size]]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.FloatTensor([np.int64(label[0]).item(), np.int64(label[1]).item(), np.int64(label[2]).item(), 
                    np.int64(label[3]).item(), np.int64(label[4]).item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def load_samples(self):
        """Load sample images from dataset."""
        numtr = 126
        numts = 60
        numvl = 14
        data_root = os.path.join(self.root, 'NIH')
        path = os.path.join(data_root, 'images')
        images = []
        labels = []
        if self.val:
            val_info = csv.reader(open(os.path.join(data_root, 'multi-val-split.csv'), 'r'))
            for count, row in enumerate(val_info):
                if count == numvl:
                    break
                image = np.repeat(np.array(Image.open(os.path.join(path, row[0])).convert('L').resize((224, 224)))[..., np.newaxis], 3, -1)
                images.append(image)
                #labels.append(row[1])
                labels.append(np.array([row[1], row[2], row[3], row[4], row[5]]))
        elif self.train:
            train_info = csv.reader(open(os.path.join(data_root, 'multi-train-split.csv'), 'r'))
            for count, row in enumerate(train_info):
                if count == numtr:
                    break
                image = np.repeat(np.array(Image.open(os.path.join(path, row[0])).convert('L').resize((224, 224)))[..., np.newaxis], 3, -1)
                images.append(image)
                #labels.append(row[1])
                labels.append(np.array([row[1], row[2], row[3], row[4], row[5]]))
        elif not self.val and not self.train:
            test_info = csv.reader(open(os.path.join(data_root, 'multi-test-split.csv'), 'r'))
            for count, row in enumerate(test_info):
                if count == numts:
                    break
                image = np.repeat(np.array(Image.open(os.path.join(path, row[0])).convert('L').resize((224, 224)))[..., np.newaxis], 3, -1)
                images.append(image)
                #labels.append(row[1])
                labels.append(np.array([row[1], row[2], row[3], row[4], row[5]]))
        images = np.asarray(images)
        labels = np.asarray(labels)
        self.dataset_size = labels.shape[0]
        return images, labels


def get_nih(train, val):
    """Get nih dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      #transforms.Normalize(
                                          #mean=params.dataset_mean,
                                          #std=params.dataset_std)])
    ])

    # dataset and data loader
    nih_dataset = NIH(root=params.data_root,
                        train=train,
                        val=val,
                        transform=pre_process)

    nih_data_loader = torch.utils.data.DataLoader(
        dataset=nih_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        drop_last = True)

    return nih_data_loader
