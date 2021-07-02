from image_slicer import slice
from PIL import Image
import torch
from torchvision import transforms

def imload_32(path):
  img = Image.open(path).convert("RGB")
  imsize = (32, 32)
  transformer = []
  transformer.append(transforms.Resize(imsize))
  t1= transforms.Compose(transformer)
  image = t1(img)
  
  return torch.from_numpy(
   np.expand_dims(np.transpose(np.array(image).astype(np.float32) / 255.0, (2, 0, 1)), 0))  # scale 0 to 1, (2,0,1) is a permutation



def imload_resize_mod8(path):
  img = Image.open(path).convert("RGB")
  print("orig size: ",img.size[0], ", ", img.size[1])
  new_size_0 = img.size[0] - img.size [0]%8
  new_size_1 = img.size[1] - img.size[1]%8
  imsize = (new_size_0, new_size_1)
  transformer = []
  transformer.append(transforms.Resize(imsize))
  t1= transforms.Compose(transformer)

  new_im = t1(img)#.unsqueeze(0)
  return new_im, new_size_0, new_size_1
'''
This function recieves the tiles of the heatmap picture, and the mean number
of iteration (Mean K, page 7 of article. No baseline value, )
'''
def calc_iterations(heatmap_tiles, mean_k = 12):
  n = len(heatmap_tiles)
  # calculating total grey value per 8x8 block:
  grey_values = []
  for tile in heatmap_tiles:
    tile.image()
    grey_value = 0
    for x in xrange(8):
      for y in xrange(8):
        r, g, b = img.getpixel((x,y))
        grey_value += r * 299.0/1000 + g * 587.0/1000 + b * 114.0/1000
    grey_values.append(grey_value)
  grey_sum = grey_values.sum()
  # calculating semantic levels (could have just calculated the number of
  # iterations, but decided to stick to the notation in the article)
  semantic_lvls =[]
  for g_val in grey_values:
    semantic_lvls.append(g_val/grey_sum)
  # With semantic levels we can calculate the number of iterations
  # Maybe the excess is better calculater as mean_k*n - 24*(num of saturated blocks)?
  iters = []
  excess = 0
  for l in semantic_lvls:
    iter = math.floor(l*mean_k*n)
    if iter > 24:
      excess += iter - 24
      iter = 24
    iters.append(iter)
  # Distribute the excess iterations among the rest of the tiles 
  # First calc the new grey_sum, without blocks with 24 iteration.
  for i in xrange(n):
    if iters[i] == 24:
      grey_sum -= grey_values[i]
      grey_values[i] = 0
  # Re-calculate the semantic levels
  for i in xrange(n):
    semantic_lvls[i] = grey_values[i]/grey_sum
  # Distribute excess iterations
  for i in xrange(n):
    iters[i] += math.floor(semantic_lvls[i]*mean_k*excess)
  return iters
  
# perhaps it's best to make another resizer which takes the mean value of the
# pic, in order to preserve the gray value of the tile
image, size0, size1 = imload_resize_mod8('heatmap_file_placeholder')
image.save('new_im.jpg')
num_blocks = (size0/8) * (size1/8)
heatmap_tiles = slice('/content/new_im.jpg', num_blocks) 
