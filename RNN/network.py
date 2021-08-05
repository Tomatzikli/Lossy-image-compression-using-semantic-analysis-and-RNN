
import torch
import torch.nn as nn
import torch.nn.functional as F
from RNN.modules import ConvLSTMCell, Sign


class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()
        self.conv = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.rnn1 = ConvLSTMCell(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, input, hidden1, hidden2, hidden3):
        x = self.conv(input)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]

        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = torch.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    def __init__(self):
        super(DecoderCell, self).__init__()

        self.conv1 = nn.Conv2d(
            32, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.rnn1 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.rnn4 = ConvLSTMCell(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.conv2 = nn.Conv2d(
            32, 3, kernel_size=1, stride=1, padding=0, bias=False)

        ###
        #down size from 32x32 to 8x8
        ###

    def forward(self, input, hidden1, hidden2, hidden3, hidden4):
        x = self.conv1(input)
        print("conv1 shape ", x.shape)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)
        print("rnn1 shape ", x.shape)

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)
        print("rnn2 shape ", x.shape)


        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)
        print("rnn3 shape ", x.shape)


        hidden4 = self.rnn4(x, hidden4)
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)
        print("rnn4 shape ", x.shape)

        x = torch.tanh(self.conv2(x)) / 2
        print("conv2 shape ", x.shape)
        return x, hidden1, hidden2, hidden3, hidden4
