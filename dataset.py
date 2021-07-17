# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
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

class Patches(data.Dataset):
    """ divide image to patches."""

    def __init__(self, image, transform=None, loader=default_loader):
        if isinstance(image, str):
            transt = transforms.ToTensor()
            image = transt(Image.open(image).convert("RGB"))
        # torch.Tensor.unfold(dimension, size, step)
        # slices the images into 8*8 size patches
        image = image.squeeze()
        patches = image.data.unfold(0, 3, 3).unfold(1, 8, 8).unfold(2, 8, 8).squeeze()
        self.transform = transform
        self.patches = patches.reshape(patches.shape[0] * patches.shape[1], 3, 8, 8)
        # print("in dataset.py patches.shape ", self.patches.shape)

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        return self.transform(self.patches[index])
        #return torch.from_numpy(np.expand_dims(np.array(img).astype(np.float32), 0))

    def __len__(self):
        return self.patches.shape[0]
