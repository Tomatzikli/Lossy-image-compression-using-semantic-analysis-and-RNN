from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

from RNN.ssim import ssim
import RNN.network as network
from global_vars import BATCH_SIZE, PATCH_SIZE


def resize(image, imsize):
    transformer = []
    transformer.append(transforms.Resize(imsize))
    trans_size = transforms.Compose(transformer)
    return trans_size(image)


def decode(batches,
           orig_size,
           output_path, model='checkpoint/decoder_epoch_00000025.pth',
           cuda=True):
    ssim_per_block = []   # calc ssim blockwise
    result = torch.tensor([])
    datasets = [batches.batch_dataset, batches.leftover_dataset]
    num_rows, num_cols = orig_size[0] // PATCH_SIZE, orig_size[1] // PATCH_SIZE
    with torch.no_grad():
        i = 0
        for patches_dataset in datasets:
            i += 1
            if i == 1:
                batch_size = BATCH_SIZE
            else:
                batch_size = 1
            for j in range(patches_dataset.__len__() // batch_size):
                content = np.load("patches/batch_{}_{}".format(i, j) + ".npz")  # extract npz file
                codes = np.unpackbits(content['codes'])  # what we saved in the encoder, save as binary numbers.
                codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1
                codes = torch.from_numpy(codes)
                iters, batch_size, channels, height, width = codes.size()
                height = height * 16
                width = width * 16

                codes = Variable(codes)

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
                        codes[iter], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                    image = image + output.data.cpu()

                result = torch.cat((result, resize(image, (PATCH_SIZE, PATCH_SIZE))), dim=0)

                # add si-ssim computation per block:
                for k in range(batch_size):
                    orig_patch = patches_dataset.__getitem__(
                        j * batch_size + k).unsqueeze(0)
                    ssim_per_block.append(ssim(orig_patch, image[k].unsqueeze(0)))

    patch_location = batches.patch_location
    # sort patches by original location in image
    result = result[torch.argsort(patch_location)]
    ssim_per_block = torch.tensor(ssim_per_block)
    ssim_per_block = ssim_per_block[torch.argsort(patch_location)]

    # combine new patches to image
    new_image = torch.tensor([])
    for i in range(num_cols):
        result_cols = torch.tensor([])
        for j in range(num_rows):
            result_cols = torch.cat((result_cols, result[i + num_cols * j]), dim=1)
        new_image = torch.cat((new_image, result_cols), dim=2)

    new_image = np.squeeze(new_image.numpy().clip(0, 1) * 255.0)
    new_image = new_image.astype(np.uint8).transpose(1, 2, 0)
    image = Image.fromarray(new_image)
    image.save(output_path)
    return ssim_per_block
