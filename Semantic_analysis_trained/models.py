import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,ConcatDataset

class CAM(nn.Module):
    def __init__(self, trained_model):
        super(CAM, self).__init__()
        self.network = ResNet(trained_model)

    def forward(self, x, topk=3):
        feature_map, output = self.network(x)
        prob, args = torch.sort(output, dim=1, descending=True)

        ## top k class probability
        topk_prob = prob.squeeze().tolist()[:topk]
        topk_arg = args.squeeze().tolist()[:topk]

        # generage class activation map
        b, c, h, w = feature_map.size()
        feature_map = feature_map.view(b, c, h * w).transpose(1,
                                                              2)  # the feature map need to be the same size as self.network.weight

        cam = torch.bmm(feature_map, self.network.fc_weight).transpose(1, 2)

        ## normalize to 0 ~ 1
        min_val, min_args = torch.min(cam, dim=2, keepdim=True)
        cam -= min_val
        max_val, max_args = torch.max(cam, dim=2, keepdim=True)
        cam /= max_val

        ## top k class activation map
        topk_cam = cam.view(1, -1, h, w)[0, topk_arg]
        topk_cam = nn.functional.interpolate(topk_cam.unsqueeze(0),
                                             (x.size(2), x.size(3)), mode='bilinear', align_corners=True).squeeze(0)
        topk_cam = torch.split(topk_cam, 1)

        return topk_prob, topk_arg, topk_cam


class ResNet(nn.Module):
    def __init__(self, network):
        super(ResNet, self).__init__()
        net_list = list(network.children())
        self.stg1 = nn.Sequential(*net_list[0])
        self.convShortcut2 = nn.Sequential(*net_list[1])
        self.conv2 = nn.Sequential(*net_list[2])
        self.ident2 = nn.Sequential(*net_list[3])
        self.convShortcut3 = nn.Sequential(*net_list[4])
        self.conv3 = nn.Sequential(*net_list[5])
        self.ident3 = nn.Sequential(*net_list[6])
        self.convShortcut4 = nn.Sequential(*net_list[7])
        self.conv4 = nn.Sequential(*net_list[8])
        self.ident4 = nn.Sequential(*net_list[9])
        self.fc_layer = net_list[-1][-1]
        self.fc_weight = nn.Parameter(self.fc_layer.weight.t().unsqueeze(0))

    def forward(self, x):
        out = self.stg1(x)

        out = F.relu(self.conv2(out) + self.convShortcut2(out))
        out = F.relu(self.ident2(out) + out)
        out = F.relu(self.ident2(out) + out)

        out = F.relu(self.conv3(out) + (self.convShortcut3(out)))
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)
        out = F.relu(self.ident3(out) + out)

        out = F.relu(self.conv4(out) + (self.convShortcut4(out)))
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)
        out = F.relu(self.ident4(out) + out)

        output = F.softmax(self.fc_layer(out.mean([2, 3])), dim=1)  # reduce over dimentions 2,3 -> GAP
        return out, output