"""
decoder.py
"""

from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from RNN import network


def resize(image, imsize):
    transformer = []
    transformer.append(transforms.Resize(imsize))
    trans_size = transforms.Compose(transformer)
    return trans_size(image)


def decode(orig_size,
           output_path, encoder_output_name, model='checkpoint/decoder_epoch_00000025.pth',
           cuda=True):
    with torch.no_grad():
        content = np.load("patch_" + encoder_output_name + ".npz")  # extract npz file
        codes = np.unpackbits(content[
                                  'codes'])  # what we saved in the encoder, save as binary numbers.
        codes = np.reshape(codes, content['shape']).astype(
            np.float32) * 2 - 1

        codes = torch.from_numpy(codes)
        iters, batch_size, channels, height, width = codes.size()
        height = height * 16  # why?
        width = width * 16

        codes = Variable(codes)

        decoder = network.DecoderCell()
        decoder.eval()

        decoder.load_state_dict(
            torch.load(model, map_location=torch.device('cpu')))

        decoder_h_1 = (Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16)),
                       Variable(
                           torch.zeros(batch_size, 512, height // 16,
                                       width // 16)))
        decoder_h_2 = (Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8)),
                       Variable(
                           torch.zeros(batch_size, 512, height // 8,
                                       width // 8)))
        decoder_h_3 = (Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4)),
                       Variable(
                           torch.zeros(batch_size, 256, height // 4,
                                       width // 4)))
        decoder_h_4 = (Variable(
            torch.zeros(batch_size, 128, height // 2, width // 2)),
                       Variable(
                           torch.zeros(batch_size, 128, height // 2,
                                       width // 2)))

        if cuda:
            decoder = decoder.cuda()

            codes = codes.cuda()

            decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
            decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
            decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
            decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

        image = torch.zeros(batch_size, 3, height, width) + 0.5

        for iter in range(iters):
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes[iter], decoder_h_1, decoder_h_2, decoder_h_3,
                decoder_h_4)
            image = image + output.data.cpu()

    # print("decoded image shape: ", image.shape)

    new_image = np.squeeze(image.numpy().clip(0, 1) * 255.0)
    new_image = new_image.astype(np.uint8).transpose(1, 2, 0)
    image = Image.fromarray(new_image)

    image.save(output_path)
