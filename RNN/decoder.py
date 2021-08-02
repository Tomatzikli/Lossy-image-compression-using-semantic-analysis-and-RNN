from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

from RNN.ssim import ssim
import math
import RNN.network as network
from global_vars import BATCH_SIZE
from global_vars import MAX_ITERATIONS

def resize(image, imsize):
    transformer = []
    transformer.append(transforms.Resize(imsize))
    trans_size = transforms.Compose(transformer)
    return trans_size(image)


def decode(batches, orig_size,
           output_path, model='checkpoint/decoder_epoch_00000025.pth',
           cuda=True):
    ssim_per_block = []
    result = torch.tensor([])
    datasets = [batches.batch_dataset, batches.leftover_dataset]
    # patch_size = 8.0
    patch_size = 32
    num_rows, num_cols = orig_size[0] // patch_size, orig_size[1] // patch_size
    with torch.no_grad():
        for patches_dataset in datasets:
            i = 1
            if i == 1:
                batch_size = BATCH_SIZE
            else:
                batch_size = 1
            print("j ", patches_dataset.__len__() // batch_size)
            for j in range(patches_dataset.__len__() // batch_size):
                print("batch = {} , batch_size = {}".format(j, batch_size))
                content = np.load("patches/batch_{}_{}".format(i, j) + ".npz")  # extract npz file
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

                result = torch.cat((result, image.squeeze()), dim=0)
                print("result size after cat: ", result.shape)

                # add si-ssim computation per block:
                for k in range(batch_size):
                    orig_patch = patches_dataset.__getitem__(
                        j * batch_size + k).unsqueeze(0)
                    ssim_per_block.append(ssim(orig_patch, image[k].unsqueeze(0)))
            i += 1

    patch_location = batches.patch_location

    print("result size = {}, locsize ={}".format(result.size(),patch_location.size()))
    print("1")
    result = result[torch.argsort(patch_location)]
    print("2")
    ssim_per_block = torch.tensor(ssim_per_block)
    ssim_per_block = ssim_per_block[torch.argsort(patch_location)]
    print("3")

    # resize to original size in order to apply ssim
    # result = resize(result, orig_size)
    print("4")

    # new_image = result.view((num_cols*patch_size, num_rows*patch_size,3))
    new_image = torch.tensor([])
    for i in range(num_cols):
        result_cols = torch.tensor([])
        for j in range(num_rows):
            result_cols = torch.cat((result_cols, result[i + num_cols * j]), dim=1)
        print("result_cols shape ", result_cols.shape)
        new_image = torch.cat((new_image, result_cols), dim=2)

    print("new_image.shape ",new_image.shape)

    new_image = np.squeeze(new_image.numpy().clip(0, 1) * 255.0)
    print("5")
    new_image = new_image.astype(np.uint8).transpose(1, 2, 0)
    print("6")
    # clip: values smaller than 0 become 0, and values larger than 1 become 1
    # from [1, 3, 32, 32] to [32, 32, 3]
    image = Image.fromarray(new_image)
    print("7")
    image.save(output_path)
    print("8")
    return ssim_per_block
