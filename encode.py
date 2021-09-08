import argparse
from heatmap_process import calc_iterations
from Semantic_analysis_trained import cam
from RNN import encoder
from torchvision import transforms
import torch
from dataset import BatchDivision
import pickle
from PIL import Image
import numpy as np
import bz2
import _pickle as cPickle


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_path', '-i', required=True, type=str, help='input image')
parser.add_argument(
    '--codes_output_directory', '-o', type=str, help='output codes', default='compressed_codes')
parser.add_argument('--cuda', '-g', help='enables cuda', default=True)
parser.add_argument(
    '--model', '-m', type=str, help='path to model', default='checkpoint/encoder_epoch_00000025.pth')

args = parser.parse_args()

def imload(path):
  image = Image.open(path).convert("RGB")
  transformer = []
  transformer.append(transforms.ToTensor())
  t1= transforms.Compose(transformer)
  image = t1(image)
  return torch.from_numpy(np.expand_dims(np.array(image).astype(np.float32), 0))


image_t = imload(args.input_path)
cam_output_path = cam.getCam(image_t, gpu=True)
iterations, semantic_level_per_block = calc_iterations(cam_output_path)
batches = BatchDivision(image_t, iterations)
print("size", batches.image_size)
encoder.encode(batches, codes_output_path=args.codes_output_directory)
path = args.codes_output_directory+'/BatchDivision' + '.pbz2'
with bz2.BZ2File(path, 'w') as f:
    cPickle.dump(batches, f)
