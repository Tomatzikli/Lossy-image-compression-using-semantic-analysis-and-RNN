import numpy as np
import torch
from torch.autograd import Variable


def encode(batches,
           model='checkpoint/encoder_epoch_00000025.pth',
           cuda=True):
    batches_data_loader = batches.batches_data_loader
    leftover_data_loader = batches.leftover_data_loader
    data_loaders = [batches_data_loader, leftover_data_loader]
    i = 1
    for loader in data_loaders:
        for batch, image in enumerate(loader):
            batch_size, input_channels, height, width = image.size()
            assert height % 32 == 0 and width % 32 == 0
            with torch.no_grad():
                image = Variable(image)
                import RNN.network as network
                encoder = network.EncoderCell()
                binarizer = network.Binarizer()
                decoder = network.DecoderCell()

                encoder.load_state_dict(torch.load(model))
                binarizer.load_state_dict(torch.load(model.replace('encoder', 'binarizer')))
                decoder.load_state_dict(torch.load(model.replace('encoder', 'decoder')))
                encoder.eval()
                binarizer.eval()
                decoder.eval()

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
                res = image - 0.5
                iters = 0
                if i == 1:
                    iters = max(batches.batch_iterations[batch * batch_size:(batch + 1) * batch_size])
                elif batches.leftover_iterations != []:
                    iters = batches.leftover_iterations[batch]
                for _ in range(iters):
                    encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                        res, encoder_h_1, encoder_h_2, encoder_h_3)

                    code = binarizer(encoded)

                    output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                        code, decoder_h_1, decoder_h_2, decoder_h_3,
                        decoder_h_4)

                    res = res - output  # residual between original image and output
                    codes.append(code.data.cpu().numpy())

                codes = (np.stack(codes).astype(np.int8) + 1) // 2
                export = np.packbits(codes.reshape(-1))
                np.savez_compressed("patches/batch_{}_{}".format(i, batch), shape=codes.shape, codes=export)
        i += 1
