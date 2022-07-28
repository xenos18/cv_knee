import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self,
                 root_dir,
                 plane,
                 train=True,
                 transform=None,
                 weights=None):

        super().__init__()
        self.plane = plane
        self.root_dir = root_dir
        self.train = train

        if self.train:
            self.folder_path = self.root_dir + "train/{0}/".format(plane)
            self.records = pd.read_csv(self.root_dir + "train.csv")
        else:
            transform = None
            self.folder_path = self.root_dir + "valid/{0}/".format(plane)
            self.records = pd.read_csv(self.root_dir + "valid.csv")

        self.records["id"] = list(self.records["id"].map(lambda i: "0" * (4 - len(str(i))) + str(i)))
        self.paths = [self.folder_path + filename + ".npy" for filename in self.records['id']]
        self.labels = {
            "acl": self.records["acl"].tolist(),
            "meniscus": self.records["meniscus"].tolist(),
            "abnormal": self.records["abnormal"].tolist(),
        }

        self.transform = transform
        if weights is None:
            self.weights = {}
            for injury in ["acl", "meniscus", "abnormal"]:
                pos = np.sum(self.labels[injury])
                neg = len(self.labels[injury]) - pos
                self.weights[injury] = neg / pos
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        images_array = np.load(self.paths[index])

        if self.transform:
            images_array = self.transform(images_array)
        else:
            images_array = np.stack((images_array,) * 3, axis=1)
            images_array = torch.FloatTensor(images_array)

        sample = {
            "image": images_array,

            "labels": torch.FloatTensor([self.labels["acl"][index],
                                         self.labels["meniscus"][index],
                                         self.labels["abnormal"][index]]),

            "weights": torch.FloatTensor([self.weights["acl"],
                                          self.weights["meniscus"],
                                          self.weights["abnormal"]])
        }

        return sample
