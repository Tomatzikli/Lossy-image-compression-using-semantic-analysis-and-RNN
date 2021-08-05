# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
from global_vars import MAX_ITERATIONS, PATCH_SIZE, BATCH_SIZE

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


def transformer_32():
    imsize = (32, 32)
    transformer = []
    transformer.append(transforms.Resize(imsize))
    return transforms.Compose(transformer)


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
    def __init__(self, image, iterations):
        image = image.squeeze()
        patches = image.data.unfold(0, 3, 3).unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE).squeeze()
        patch_by_iters = {}
        location_by_iters = {}
        self.patch_location = []
        batch_patches = torch.tensor([])
        self.batch_locations = []
        self.batch_iterations = []
        self.leftover_iterations = []
        leftover_patches = torch.tensor([])
        num_rows = patches.shape[0]
        num_cols = patches.shape[1]
        for i in range(num_rows):
            for j in range(num_cols):
                patch_iters = iterations[i * num_cols + j]
                if patch_iters not in patch_by_iters:
                    location_by_iters[patch_iters] = []
                    patch_by_iters[patch_iters] = torch.tensor([])
                patch_by_iters[patch_iters] = torch.cat((patch_by_iters[patch_iters], patches[i][j].unsqueeze(0)), dim=0)
                location_by_iters[patch_iters].append(i * num_cols + j)
        for i in range(0, MAX_ITERATIONS+1):
            if i in patch_by_iters:
                batch_patches = torch.cat((batch_patches, patch_by_iters[i]), dim=0)
                self.patch_location += location_by_iters[i]
                for _ in range(len(patch_by_iters[i])):
                    self.batch_iterations.append(i)

        self.patch_location = torch.tensor(self.patch_location)
        leftover_index = len(batch_patches) % BATCH_SIZE
        if leftover_index != 0:
            leftover_patches = batch_patches[-leftover_index:]
            self.leftover_dataset = Patches(leftover_patches, transform=transformer_32())
            self.leftover_iterations = self.batch_iterations[-leftover_index:]
            self.batch_iterations = self.batch_iterations[:-leftover_index]
            batch_patches = batch_patches[:-leftover_index]
            self.batch_dataset = Patches(batch_patches, transform=transformer_32())
        else:
            self.leftover_dataset = Patches(leftover_patches,
                                            transform=transformer_32())
            self.batch_dataset = Patches(batch_patches,
                                         transform=transformer_32())
        print("leftover shape: ", leftover_patches.shape)
        print("batch_dataset size = {} , BATCH_SIZE = {}".format(self.batch_dataset.__len__(), BATCH_SIZE))
        self.batches_data_loader = DataLoader(dataset=self.batch_dataset,
                                              batch_size=BATCH_SIZE,
                                              num_workers=4,
                                              pin_memory=True)
        self.leftover_data_loader = DataLoader(dataset=self.leftover_dataset,
                                               batch_size=1,
                                               num_workers=4, pin_memory=True)


class Patches(data.Dataset):
    """ divide image to patches."""

    def __init__(self, image, transform=None, loader=default_loader):
        self.patches = image
        # print("{}".format(self.patches.size()))
        # print("in dataset.py patches.shape ", self.patches.shape)

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        return self.transform(self.patches[index])
        # return torch.from_numpy(np.expand_dims(np.array(img).astype(np.float32), 0))

    def __len__(self):
        return self.patches.shape[0]
