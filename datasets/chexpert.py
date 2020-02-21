import os
import csv
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

import params


class Chexpert(data.Dataset):

    def __init__(self, root, train=True, transform=None, download=False):
        """Init chexpert dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.dataset_size = None


        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size]]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        #self.train_data = self.train_data.transpose(
            #(2, 0, 1))  # convert to HWC
        

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
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def load_samples(self):
        """Load sample images from dataset."""
        data_root = os.path.join(self.root, 'CheXpert-v1.0-small')
        train_info = csv.reader(open(os.path.join(data_root, 'trainer.csv'), 'r'))
        valid_info = csv.reader(open(os.path.join(data_root, 'valider.csv'), 'r'))
        images = []
        labels = []
        counter = 0
        if self.train:
            for row in train_info:
                filename = os.path.join(self.root, row[0])
                img = Image.open(filename)
                image = np.array(img)
                #image = np.array([[[s,s,s] for s in r] for r in image])
                #print(image.shape)
                #print(image)
                label = row[1]
                images.append(image)
                labels.append(label)
                counter += 1
                if counter == 20:
                    break
            images = np.asarray(images)
            labels = np.asarray(labels)
            self.dataset_size = labels.shape[0]
        else:
            for row in valid_info:
                filename = os.path.join(self.root, row[0])
                img = Image.open(filename)
                image = np.array(img)
                #image = np.array([[[s,s,s] for s in r] for r in image])
                #print(image.shape)
                label = row[1]
                images.append(image)
                labels.append(label)
                counter += 1
                if counter == 10:
                    break
            images = np.asarray(images)
            labels = np.asarray(labels)
            self.dataset_size = labels.shape[0]
        return images, labels


def get_chexpert(train):
    """Get chexpert dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      #transforms.Normalize(
                                          #mean=params.dataset_mean,
                                          #std=params.dataset_std)])
    ])

    # dataset and data loader
    chexpert_dataset = Chexpert(root=params.data_root,
                        train=train,
                        transform=pre_process,
                        download=True)

    chexpert_data_loader = torch.utils.data.DataLoader(
        dataset=chexpert_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return chexpert_data_loader
