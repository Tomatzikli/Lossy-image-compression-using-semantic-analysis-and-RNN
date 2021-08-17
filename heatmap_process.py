from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import math
from global_vars import PATCH_SIZE, MIN_ITERATIONS


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
'''
def calc_iterations(path, mean_k = 12):
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
  semantic_lvls =[]
  for g_val in grey_values:
    semantic_lvls.append(g_val/grey_sum)
  # With semantic levels we can calculate the number of iterations
  iters = []
  for _ in range(n):
    iters.append(MIN_ITERATIONS)
  excess = 0
  i = 0
  total_iters = mean_k * n - MIN_ITERATIONS * n
  for l in semantic_lvls:
    iter = math.floor(l*total_iters)
    if iter > 24 - MIN_ITERATIONS:
      excess += iter - 24 
      iter = 24
    iters[i] += iter
  # Distribute the excess iterations among the rest of the tiles
  # First calc the new grey_sum, without blocks with 24 iteration.
  for i in range(n):
    if iters[i] == 24:
      grey_sum -= grey_values[i]
      grey_values[i] = 0
  # Re-calculate the semantic levels
  for i in range(n):
    semantic_lvls[i] = grey_values[i]/grey_sum
  # Distribute excess iterations
  for i in range(n):
    iters[i] += math.floor(semantic_lvls[i]*excess)
    # Each tile must be passed at least once
    if iters[i] == 0:
        iters[i] = 1
    iters = np.array(iters).clip(1,24)
  return iters, torch.tensor(semantic_lvls)
