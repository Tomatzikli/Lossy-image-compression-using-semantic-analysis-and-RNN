import argparse
import torch
from heatmap_process import calc_iterations
# from Semantic_analysis_trained import cam
from Semantic_analysis import cam
from RNN import encoder, decoder
from RNN import metric
from torchvision import transforms
import os
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BatchDivision

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_directory', '-i', required=True, type=str, help='input image directory')
parser.add_argument(
    '--output_directory', '-o', type=str, help='output codes', default='compressed_output')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda', default=True)

parser.add_argument(
    '--model', '-m', type=str, help='path to model', default='checkpoint/encoder_epoch_00000025.pth')

args = parser.parse_args()

print("args: ", args)

images_set = dataset.ImageFolder(root=args.input_directory, transform=transforms.ToTensor())
images_loader = DataLoader(dataset=images_set, batch_size=1, num_workers=4)
filenames = sorted(os.listdir(args.input_directory))
for batch, image_t in tqdm(enumerate(images_loader)):
    cam_output_path = cam.getCam(image_t, gpu=True)
    iterations, semantic_level_per_block = calc_iterations(cam_output_path)
    batches = BatchDivision(image_t, iterations)
    encoder.encode(batches)
    output_path = os.path.join(args.output_directory, filenames[batch])
    decoder.decode(batches, orig_size=(image_t.shape[2], image_t.shape[3]), output_path=output_path)
