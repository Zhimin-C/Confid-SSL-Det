from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision
import json
import numpy as np
import os
import sys
import math
from scannet.model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()
BASE_DIR = '/home/scratch1link/Semi-Vit/det/scannet/3DIoUMatch/scannet/scannet_train_detection_data'
ROOT_DIR = BASE_DIR
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label
        self.data_path = ROOT_DIR

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        self.scan_namse = dataset.scan_names

        weights = np.ones(len(self.scan_namse))*0.1

        z = open("dict_avgconf.txt", "r")
        k = z.read()
        dict_avgconf = json.loads(k)
        z.close()
        
        z = open("current_epoch.txt", "r")
        k = z.read()
        current_epoch = json.loads(k)
        z.close()
        
        soft_ratio = math.exp(-3 * (1 - min(current_epoch/1000.0, 1))**3)

        for index in range(len(self.scan_namse)):
            scan_name = self.scan_namse[index]
            bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')  # K,8
            min_confid = 1
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                current_class = bbox[6]
                semantic_class = np.where(DC.nyu40ids == current_class)[0]                
                
                if min_confid > dict_avgconf['%d' %semantic_class]:
                    min_confid = dict_avgconf['%d' %semantic_class]
            if min_confid!=1:
                weights[index] = 1 - soft_ratio*min_confid

#         weights[weights==1] = min(weights)

            # semantic_class = bbox[7]

        # distribution of classes in the dataset

        self.weights = torch.DoubleTensor(weights.tolist())

        # print(1)

            # weights = 1.0 / label_to_count[df["label"]]
        #
        # self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.label.squeeze(-1)
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
