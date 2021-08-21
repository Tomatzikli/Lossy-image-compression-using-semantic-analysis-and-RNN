import numpy as np
from PIL import Image
from matplotlib import cm

import torchvision
import torchvision.transforms as transforms


def _normalizer(denormalize=False):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    if denormalize:
        MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
        STD = [1 / std for std in STD]

    return transforms.Normalize(mean=MEAN, std=STD)


def _transformer(imsize=None, cropsize=None):
    transformer = []
    if imsize:
        transformer.append(transforms.Resize(imsize))
    if cropsize:
        transformer.append(transforms.CenterCrop(cropsize))
    transformer.append(transforms.ToTensor())
    transformer.append(_normalizer())
    return transforms.Compose(transformer)

def imload_tensor(image_t):
    transformer = _normalizer()
    return transformer(image_t)

def imload_8(path):
  img = Image.open(path).convert("RGB")
  new_size_0 = img.size[0] - img.size[0]%8
  new_size_1 = img.size[1] - img.size[1]%8
  imsize = (new_size_0, new_size_1)
  transformer = []
  transformer.append(transforms.Resize(imsize))
  transformer.append(transforms.ToTensor())
  transformer.append(_normalizer())
  t1= transforms.Compose(transformer)

  new_im = t1(img).unsqueeze(0)
  return new_im


def extract_blocks(a, blocksize, keep_as_view=False):
    M,N = a.shape
    b0, b1 = blocksize
    if keep_as_view==0:
        return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2).reshape(-1,b0,b1)
    else:
        return a.reshape(M//b0,b0,N//b1,b1).swapaxes(1,2)


def imload(path, imsize=None, cropsize=None):
    transformer = _transformer(imsize=imsize, cropsize=cropsize)
    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)


def imsave(path, tensor):
    denormalize = _normalizer(denormalize=True)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)
    torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
    return None


def imshow(tensor):
    denormalize = _normalizer(denormalize=True)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(denormalize(tensor.squeeze()))
    image = torchvision.transforms.functional.to_pil_image(tensor)
    return image


def array_to_cam(arr):
    cam_pil = Image.fromarray(np.uint8(cm.gist_earth(arr) * 255)).convert("RGB")
    return cam_pil


def blend(image1, image2, alpha=0.75):
    return Image.blend(image1, image2, alpha)