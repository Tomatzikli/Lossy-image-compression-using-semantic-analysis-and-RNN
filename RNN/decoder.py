from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from RNN.ssim import ssim

def resize(image, imsize):
    transformer = []
    transformer.append(transforms.Resize(imsize))
    trans_size = transforms.Compose(transformer)
    return trans_size(image)

def decode(num_rows, num_cols, patches, iterations, orig_size_0, orig_size_1,
           model='checkpoint/decoder_epoch_00000025.pth', cuda=True):
    ssim_per_block = []
    result = torch.tensor([])
    with torch.no_grad():
        for i in range(num_rows):
          row_tensor = torch.tensor([])
          for j in range(num_cols):
            print("iteration {},{} out of ".format(i,j), num_rows," ",num_cols)
            content = np.load("patch_{}_{}".format(i,j)+".npz")   # extract npz file
            codes = np.unpackbits(content['codes'])  # what we saved in the encoder, save as binary numbers.
            codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1

            codes = torch.from_numpy(codes)
            iters, batch_size, channels, height, width = codes.size()
            height = height * 16   # why?
            width = width * 16

            codes = Variable(codes)

            import RNN.network as network
            decoder = network.DecoderCell()
            decoder.eval()

            decoder.load_state_dict(torch.load(model, map_location=torch.device('cpu')))

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

            ##############
            cuda = False
            #############

            if cuda:
                decoder = decoder.cuda()

                codes = codes.cuda()

                decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
                decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
                decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
                decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

            image = torch.zeros(1, 3, height, width) + 0.5

            for iters in range(min(iterations[i][j], codes.size(0))):
                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                image = image + output.data.cpu()

            # print("image size before cat: ", image.shape) 1,3,32,32
            row_tensor = torch.cat((row_tensor,image.squeeze()), dim=2)
            # print("rowtensor size after cat: ", row_tensor.shape)
            # add si-ssim computation per block:
            m1 = patches[i][j].unsqueeze(0)
            m2 = resize(image, (8,8))
            ## זה print("sizes: ", m1.shape, m2.shape)
            ssim_per_block.append(ssim(m1, m2))
          result = torch.cat((result, row_tensor), dim=1)
          # print("result after cat: ", result.shape)

    imsize = (orig_size_0, orig_size_1)  # resize to original size in order to apply ssim
    result = resize(result, imsize)
    new_image =  np.squeeze(result.numpy().clip(0, 1) * 255.0)
    new_image = new_image.astype(np.uint8).transpose(1, 2, 0)
    # clip: values smaller than 0 become 0, and values larger than 1 become 1
    # from [1, 3, 32, 32] to [32, 32, 3]
    image = Image.fromarray(new_image)

    image.save('checkpoint/watch-join.png')
    return torch.tensor(ssim_per_block)
