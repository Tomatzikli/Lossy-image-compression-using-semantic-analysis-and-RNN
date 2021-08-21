from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import math
from global_vars import PATCH_SIZE, MIN_ITERATIONS, MAX_ITERATIONS


def imload(path):
    img = Image.open(path).convert("RGB")
    imsize = (PATCH_SIZE, PATCH_SIZE)
    transformer = []
    transformer.append(transforms.Resize(imsize))
    t1 = transforms.Compose(transformer)
    image = t1(img)
    return torch.from_numpy(
        np.expand_dims(np.transpose(np.array(image).astype(np.float32) / 255.0, (2, 0, 1)), 0))


def imload_resize_mod8(path):
    img = Image.open(path).convert("RGB")
    new_size_0 = img.size[0] - img.size[0] % PATCH_SIZE
    new_size_1 = img.size[1] - img.size[1] % PATCH_SIZE
    imsize = (new_size_0, new_size_1)
    transformer = []
    transformer.append(transforms.Resize(imsize))
    t1 = transforms.Compose(transformer)
    new_im = t1(img)
    return new_im, new_size_0, new_size_1


'''
This function recieves the tiles of the heatmap picture, and the mean number
of iteration (Mean K, page 7 of article. No baseline value, )
gives each tile num of iterations s.t the mean is mean_k, the minimum is MIN_ITERATIONS, the maximum is MAX_ITERATIONS
'''
def calc_iterations(path, mean_k=6):
  transt = transforms.ToTensor()
  image = transt(Image.open(path).convert("RGB"))
  image = image.squeeze()
  heatmap_tiles = image.data.unfold(0, 3, 3).unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE).squeeze()
  num_rows = heatmap_tiles.shape[0]
  num_cols = heatmap_tiles.shape[1]
  n = num_rows * num_cols
  # calculating total grey value per 8x8 block:
  grey_values = []
  for i in range(num_rows):
    for j in range(num_cols):
      block = heatmap_tiles[i][j]
      grey_value = 0
      for x in range(PATCH_SIZE):
        for y in range(PATCH_SIZE):
          r, g, b = block[0][x][y], block[1][x][y], block[2][x][y]
          grey_value += r * 299.0/1000 + g * 587.0/1000 + b * 114.0/1000
      grey_values.append(grey_value)
  grey_sum = sum(grey_values)
  # calculating semantic levels (could have just calculated the number of
  # iterations, but decided to stick to the notation in the article)
  semantic_level = []
  for g_val in grey_values:
    semantic_level.append(g_val/grey_sum)
  semantic_level = np.array(semantic_level)
  # The total number of iterations we wan't to divide between the patches
  tot_iterations = n * mean_k
  # first give all the patches the minimum num of iterations
  iter = np.full_like(semantic_level, fill_value=MIN_ITERATIONS)
  # divide the rest of the iterations according to the semantic level
  left_iterations = tot_iterations - iter.sum()
  iter = iter + np.floor(semantic_level * left_iterations)
  # update the semantic level, divide the excess iterations to the patches with less than max-iterations.
  iter[iter > MAX_ITERATIONS] = MAX_ITERATIONS
  left_iterations = tot_iterations - iter.sum()
  updated_slevel = semantic_level * (iter < MAX_ITERATIONS).astype(int)
  updated_slevel /= updated_slevel.sum()
  iter = iter + np.floor(updated_slevel * left_iterations)
  left_iterations = tot_iterations - iter.sum()
  # divide the excess iteration directly between the patches
  # prefer those with bigger num of iteration,  thats less than max-iteration.
  for i in range(int(left_iterations)):
    iter[(iter * (iter < MAX_ITERATIONS)).argmax()] += 1
  iter = np.array(iter).clip(1, MAX_ITERATIONS)
  print("mean iters ", sum(iter)/len(iter))
  return iter, torch.tensor(semantic_level)


