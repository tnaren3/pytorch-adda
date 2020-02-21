import os
import csv
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

import params


class NIH(data.Dataset):

    def __init__(self, root, train=True, transform=None, download=False):
        """Init NIH dataset."""
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
        data_root = os.path.join(self.root, 'NIH')
        train_info = csv.reader(open(os.path.join(data_root, 'trainer.csv'), 'r'))
        valid_info = csv.reader(open(os.path.join(data_root, 'valider.csv'), 'r'))
        images = []
        labels = []
        counter = 0
        if self.train:
            for row in train_info:
                if counter < 4999:
                    path = os.path.join(data_root, 'images_001/images')
                elif counter < 14999:
                    path = os.path.join(data_root, 'images_002/images')
                elif counter < 24999:
                    path = os.path.join(data_root, 'images_003/images')
                elif counter < 34999:
                    path = os.path.join(data_root, 'images_004/images')
                elif counter < 44999:
                    path = os.path.join(data_root, 'images_005/images')
                elif counter < 54999:
                    path = os.path.join(data_root, 'images_006/images')
                elif counter < 64999:
                    path = os.path.join(data_root, 'images_007/images')
                elif counter < 74999:
                    path = os.path.join(data_root, 'images_008/images')
                elif counter < 84999:
                    path = os.path.join(data_root, 'images_009/images')
                elif counter < 94999:
                    path = os.path.join(data_root, 'images_010/images')
                elif counter < 104999:
                    path = os.path.join(data_root, 'images_011/images')
                filename = os.path.join(path, row[0])
                img = Image.open(filename)
                image = np.array(img.convert('L'))
                #image = image[:,:,0]
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
            path = os.path.join(data_root, 'images_012/images')
            for row in valid_info:
                filename = os.path.join(path, row[0])
                img = Image.open(filename)
                image = np.array(img.convert('L'))
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


def get_nih(train):
    """Get nih dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      #transforms.Normalize(
                                          #mean=params.dataset_mean,
                                          #std=params.dataset_std)])
    ])

    # dataset and data loader
    nih_dataset = NIH(root=params.data_root,
                        train=train,
                        transform=pre_process,
                        download=True)

    nih_data_loader = torch.utils.data.DataLoader(
        dataset=nih_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return nih_data_loader
