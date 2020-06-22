import os
import csv
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

import params



class COVID19_Dataset(Dataset):
    """
    COVID-19 image data collection
    Dataset: https://github.com/ieee8023/covid-chestxray-dataset
    
    Paper: https://arxiv.org/abs/2003.11597
    """
    
    def __init__(self, 
                 imgpath=os.path.join(thispath, "covid-chestxray-dataset", "images"), 
                 csvpath=os.path.join(thispath, "covid-chestxray-dataset", "metadata.csv"), 
                 views=["PA", "AP"],
                 transform=None, 
                 data_aug=None, 
                 nrows=None, 
                 seed=0,
                 pure_labels=False, 
                 unique_patients=True):

        super(COVID19_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = views
        
        # defined here to make the code easier to read
        pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus", "Pneumocystis", "Klebsiella", "Chlamydophila", "Legionella", "Influenza", "Mycoplasma", "Varicella", "Viral", "Bacterial", "Fungal", "Lipoid","E.Coli"]
        
        self.pathologies = ["Pneumonia","No Finding"] + pneumonias
        self.pathologies = sorted(self.pathologies)

        mapping = dict()
        mapping["Pneumonia"] = pneumonias
        mapping["Viral"] = ["COVID-19", "SARS", "MERS", "Influenza", "Varicella"]
        mapping["Bacterial"] = ["Streptococcus", "Klebsiella", "Chlamydophila", "Legionella", "Mycoplasma","E.Coli"]
        mapping["Fungal"] = ["Pneumocystis"]
        
        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Keep only the frontal views.
        #idx_pa = self.csv["view"].isin(["PA", "AP", "AP Supine"])
        idx_pa = self.csv["view"].isin(self.views)
        self.csv = self.csv[idx_pa]
        
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["finding"].str.contains(pathology)
            if pathology in mapping:
                for syn in mapping[pathology]:
                    #print("mapping", syn)
                    mask |= self.csv["finding"].str.contains(syn)
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={} views={}".format(len(self), self.views)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['filename'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        #print(img_path)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]                    
                               
        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)
            
        return {"img":img, "lab":self.labels[idx], "idx":idx}