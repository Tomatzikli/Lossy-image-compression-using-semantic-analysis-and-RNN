import argparse
from RNN import decoder
import os
import pickle
import bz2
import _pickle as cPickle


parser = argparse.ArgumentParser()
parser.add_argument(
    '--codes_input_directory', '-i', type=str, help='input image directory', default='compressed_codes')
parser.add_argument(
    '--output_directory', '-o', type=str, help='output codes', default='compressed_output')
parser.add_argument('--cuda', '-g', help='enables cuda', default=True)
parser.add_argument(
    '--model', '-m', type=str, help='path to model', default='checkpoint/encoder_epoch_00000025.pth')

args = parser.parse_args()

path = args.codes_input_directory+'/BatchDivision' + '.pbz2'
batches = bz2.BZ2File(path, 'rb')
batches = cPickle.load(batches)
output_path = os.path.join(args.output_directory, "result.jpg")
decoder.decode(batches, orig_size=(batches.image_size[1], batches.image_size[2]),
               output_path=output_path, codes_input_path=args.codes_input_directory)

