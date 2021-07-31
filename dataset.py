# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import main
import numpy as np

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, loader=default_loader):
        images = []
        for filename in os.listdir(root):
            if is_image_file(filename):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename = self.imgs[index]
        try:
            img = self.loader(os.path.join(self.root, filename))
        except:
            return torch.zeros((3, 32, 32))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


class BatchDivision():
    def __init__(self, patches, iterations):

        patch_by_iters = {}
        location_by_iters = {}
        self.patch_location = []
        batch_patches = []
        self.batch_locations = []
        self.batch_iterations = []
        self.leftover_iterations = []
        leftover_patches = []
        num_rows = patches.shape[0]
        num_cols = patches.shape[1]
        for i in range(num_rows):
            for j in range(num_cols):
                patch_iters = iterations[i * self.num_cols + j]
                if patch_by_iters[patch_iters] == None:
                    location_by_iters[patch_iters] = []
                    patch_by_iters[patch_iters] = []
                patch_by_iters[patch_iters].add(patches[i][j])
                location_by_iters[patch_iters].add(i * num_cols + j)
        for i in range(0, main.MAX_ITERATIONS):
            if patch_by_iters[i] != None:
                self.batch_patches.add(patch_by_iters[i])
                self.patch_location.add(location_by_iters[i])
                for _ in range(len(patch_by_iters[i])):
                    self.batch_iterations.add(i)

        leftover_index = len(batch_patches) % main.BATCH_SIZE
        leftover_patches = batch_patches[-leftover_index:]
        self.leftover_dataset = Patches(leftover_patches)
        self.leftover_iterations = self.batch_iterations[-leftover_index:]
        self.batch_iterations = self.batch_iterations[:-leftover_index]
        batch_patches = batch_patches[:-leftover_index]
        self.batch_dataset = Patches(batch_patches)
        self.batches_data_loader = DataLoader(dataset=self.batch_dataset,
                                              batch_size=main.BATCH_SIZE,
                                              num_workers=4,
                                              pin_memory=True)
        self.leftover_data_loader = DataLoader(dataset=self.leftover_dataset,
                                               batch_size=1,
                                               num_workers=4, pin_memory=True)


class Patches(data.Dataset):
    """ divide image to patches."""

    def __init__(self, image, transform=None, loader=default_loader):
        self.patches = image
        # print("in dataset.py patches.shape ", self.patches.shape)

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        return self.transform(self.patches[index])
        # return torch.from_numpy(np.expand_dims(np.array(img).astype(np.float32), 0))

    def __len__(self):
        return self.patches.shape[0]
