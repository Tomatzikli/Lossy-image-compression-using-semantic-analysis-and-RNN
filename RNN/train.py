"""
train.py
"""

import time
import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS

from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
import torchvision
from torchvision.datasets import CIFAR100

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', '-N', type=int, default=64, help='batch size')
parser.add_argument('--train', '-f', type=str, help='folder of training images', default='/content/cifar-10-batches-py')
parser.add_argument('--max-epochs', '-e', type=int, default=50, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
args = parser.parse_args("")

train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)),
    transforms.ToTensor(),
])

train_set = CIFAR100(download=True, root="./data", transform=train_transform)
validation_set = CIFAR100(download=True, train=False, root="./data", transform=train_transform)

train_loader = data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
print('total images: {}; total batches: {}'.format(len(train_set), len(train_loader)))
validation_loader = data.DataLoader(
    dataset=validation_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

## load networks on GPU

encoder = EncoderCell().cuda()
binarizer = Binarizer().cuda()
decoder = DecoderCell().cuda()

solver = optim.Adam(
    [
        {
            'params': encoder.parameters()  ## generator of parameters?
        },
        {
            'params': binarizer.parameters()
        },
        {
            'params': decoder.parameters()
        },
    ],
    lr=args.lr)


def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load('checkpoint/encoder_{}_{:08d}.pth'.format(s, epoch)))
    binarizer.load_state_dict(
        torch.load('checkpoint/binarizer_{}_{:08d}.pth'.format(s, epoch)))
    decoder.load_state_dict(
        torch.load('checkpoint/decoder_{}_{:08d}.pth'.format(s, epoch)))


def save(index, epoch=True):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    # torch.save(encoder.state_dict(), 'checkpoint/encoder_{}_{:08d}.pth'.format(
    #     s, index))
    # torch.save(binarizer.state_dict(),
    #            'checkpoint/binarizer_{}_{:08d}.pth'.format(s, index))
    # torch.save(decoder.state_dict(), 'checkpoint/decoder_{}_{:08d}.pth'.format(
    #     s, index))

    torch.save(encoder.state_dict(),
               'checkpoint/encoder_{}_{:08d}.pth'.format(s, index))
    torch.save(binarizer.state_dict(),
               'checkpoint/binarizer_{}_{:08d}.pth'.format(s, index))
    torch.save(decoder.state_dict(),
               'checkpoint/decoder_{}_{:08d}.pth'.format(s, index))


def evaluate_model(data, encoder, binarizer, decoder):
    encoder.eval()
    binarizer.eval()
    decoder.eval()
    ## init lstm state
    encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                   Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
    encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
    encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))

    decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
    decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
    decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                   Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
    decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                   Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))

    patches = Variable(data.cuda())
    losses = []
    res = patches - 0.5

    for _ in range(args.iterations):
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
            res, encoder_h_1, encoder_h_2, encoder_h_3)

        codes = binarizer(encoded)

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        res = res - output
        losses.append(res.abs().mean())

    loss = sum(losses) / args.iterations

    return loss


# ======= resume
resume(9)
scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100],
                           gamma=0.5)  # https://www.programmersought.com/article/35594904715/

last_epoch = 9
if args.checkpoint:
    resume(args.checkpoint)
    last_epoch = args.checkpoint
    scheduler.last_epoch = last_epoch - 1

train_losses = []
validation_losses = []

for epoch in range(last_epoch + 1, args.max_epochs + 1):
    print("epoch {}".format(epoch))
    loss_all_batch = []
    for batch, data_pair in enumerate(train_loader):
        data = data_pair[0]  # I added#
        batch_t0 = time.time()

        ## init lstm state
        encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))

        decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
        decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
        decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
        decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                       Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))

        patches = Variable(data.cuda())

        solver.zero_grad()

        losses = []

        res = patches - 0.5

        bp_t0 = time.time()

        for _ in range(args.iterations):
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3)

            codes = binarizer(encoded)

            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

            res = res - output
            losses.append(res.abs().mean())

        bp_t1 = time.time()

        loss = sum(losses) / args.iterations
        loss.backward()

        solver.step()
        scheduler.step()

        batch_t1 = time.time()

        # loss_all_batch.append(loss.data[0])
        loss_all_batch.append(loss.data.item())
        # print(
        # '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
        # format(epoch, batch + 1,
        # len(train_loader), loss.data[0], bp_t1 - bp_t0, batch_t1 -batch_t0))
        #          len(train_loader), loss.data, bp_t1 - bp_t0, batch_t1 -batch_t0))
        # print(('{:.4f} ' * args.iterations +'\n').format(* [l.data[0] for l in losses]))
        # print(('{:.4f} ' * args.iterations +'\n').format(* [l.data for l in losses]))
        index = (epoch - 1) * len(train_loader) + batch

        ## save checkpoint every 500 training steps
        if index % 500 == 0:
            save(0, False)

    train_loss_epoch = sum(loss_all_batch) / len(loss_all_batch)
    print("train_loss_epoch {} ".format(epoch), train_loss_epoch)
    train_losses.append(train_loss_epoch)

    loss_all_val_batch = []
    for batch, data_pair in enumerate(validation_loader):
        val_loss = evaluate_model(data_pair[0], encoder, binarizer, decoder)
        loss_all_val_batch.append(val_loss.data.item())

    val_loss_epoch = sum(loss_all_val_batch) / len(loss_all_val_batch)
    print("val_loss_epoch {} ".format(epoch), val_loss_epoch)
    validation_losses.append(val_loss_epoch)

    save(epoch)
    df = pd.DataFrame()
    df["train_loss"] = train_losses
    df["val_loss"] = validation_losses
    df.to_excel('loss_results_epoch{}'.format(epoch) + '.xlsx')

print(train_losses)
print(validation_losses)
