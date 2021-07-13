import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

def imload_32(tensor):
  imsize = (32, 32)
  transformer = []
  transformer.append(transforms.Resize(imsize))
  t1= transforms.Compose(transformer)
  image = t1(tensor)
  return torch.from_numpy(np.expand_dims(np.array(image).astype(np.float32), 0))


def encode(patches, iterations, model='checkpoint/encoder_epoch_00000025.pth', cuda=True):
    num_rows = patches.shape[0]
    num_cols = patches.shape[1]
    for i in range(num_rows):
      for j in range(num_cols):
        image = imload_32(patches[i][j])
        batch_size, input_channels, height, width = image.size()
        assert height % 32 == 0 and width % 32 == 0

        with torch.no_grad():
          image = Variable(image)
          import RNN.network as network
          encoder = network.EncoderCell()
          binarizer = network.Binarizer()
          decoder = network.DecoderCell()
          encoder.eval()
          binarizer.eval()
          decoder.eval()

          encoder.load_state_dict(torch.load(model))
          binarizer.load_state_dict(torch.load(model.replace('encoder', 'binarizer')))
          decoder.load_state_dict(torch.load(model.replace('encoder', 'decoder')))

          encoder_h_1 = (Variable(
              torch.zeros(batch_size, 256, height // 4, width // 4)),
                        Variable(
                            torch.zeros(batch_size, 256, height // 4, width // 4)))
          encoder_h_2 = (Variable(
              torch.zeros(batch_size, 512, height // 8, width // 8)),
                        Variable(
                            torch.zeros(batch_size, 512, height // 8, width // 8)))
          encoder_h_3 = (Variable(
              torch.zeros(batch_size, 512, height // 16, width // 16)),
                        Variable(
                            torch.zeros(batch_size, 512, height // 16, width // 16)))

          decoder_h_1 = (Variable(
              torch.zeros(batch_size, 512, height // 16, width // 16)),
                        Variable(
                            torch.zeros(batch_size, 512, height // 16, width // 16)))
          decoder_h_2 = (Variable(
              torch.zeros(batch_size, 512, height // 8, width // 8)),
                        Variable(
                            torch.zeros(batch_size, 512, height // 8, width // 8)))
          decoder_h_3 = (Variable(
              torch.zeros(batch_size, 256, height // 4, width // 4)),
                        Variable(
                            torch.zeros(batch_size, 256, height // 4, width // 4)))
          decoder_h_4 = (Variable(
              torch.zeros(batch_size, 128, height // 2, width // 2)),
                        Variable(
                            torch.zeros(batch_size, 128, height // 2, width // 2)))

          if cuda:
              encoder = encoder.cuda()
              binarizer = binarizer.cuda()
              decoder = decoder.cuda()

              image = image.cuda()

              encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
              encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
              encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

              decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
              decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
              decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
              decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

          codes = []
          res = image - 0.5   # ? why, what
          for iters in range(max(iterations[i][j],1)):
              encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                  res, encoder_h_1, encoder_h_2, encoder_h_3)

              code = binarizer(encoded)

              output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                  code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

              res = res - output   ## diff between original image and output?
              codes.append(code.data.cpu().numpy()) # what is this?
              ## print('Iter: {:02d}; Loss: {:.06f}'.format(iters, res.data.abs().mean()))

          codes = (np.stack(codes).astype(np.int8) + 1) // 2
          export = np.packbits(codes.reshape(-1))   # Packs the elements of a binary-valued array into bits in a uint8 array. 2048 -> 256
          np.savez_compressed("patches/patch_{}_{}".format(i,j), shape=codes.shape, codes=export)   # the last two is a dic

    return num_rows, num_cols