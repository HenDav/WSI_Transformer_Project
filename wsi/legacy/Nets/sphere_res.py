# -*- coding:utf-8 -*-

import torch.nn as nn
import torch
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import os


def tensor_to_gpu(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


def tensor_to_cpu(tensor, is_cuda):
    if is_cuda:
        return tensor.cpu()
    else:
        return tensor


cfg = {
    12: [1, 1, 1, 1, 1],
    18: [1, 2, 2, 2, 1],
    20: [1, 2, 4, 1, 1],
    28: [1, 3, 6, 1, 1],
    36: [2, 4, 8, 2, 1],
    64: [3, 8, 16, 3, 1],
}

block2channels = {
    0: 16,
    1: 32,
    2: 64,
    3: 128,
    4: 256
}


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, instance_norm=False):
        super(ResBlock, self).__init__()
        self.conv2d_1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels) if not instance_norm else nn.InstanceNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2d_2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels) if not instance_norm else nn.InstanceNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2d_3 = conv3x3(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels) if not instance_norm else nn.InstanceNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        net = self.conv2d_1(x)
        net = self.bn1(net)
        net = self.relu1(net)

        net = self.conv2d_2(net)
        net = self.bn2(net)
        net = self.relu2(net)

        net = self.conv2d_3(net)
        net = self.bn3(net)
        net = self.relu3(net)

        if x.size(1) < net.size(1):
            x = F.pad(x, x.view(1) - net.view(1))

        # if the num of channels of the input is larger than the outputs' - don't use the residual connection
        if x.size(1) > net.size(1):
            pass
        else:
            net = net + x
        return net


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, instance_norm=False):
        super(DownSampleBlock, self).__init__()
        self.conv2d = conv3x3(in_channels, out_channels, stride=2)
        self.relu = nn.ReLU()
        if not instance_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv2d(x)))


class UpSampleBlock(nn.Module):
    def __init__(self):
        super(UpSampleBlock, self).__init__()
        self.us = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        return self.us(x)


class SphereFaceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repetitions, keep_res=False, instance_norm=False):
        super(SphereFaceBlock, self).__init__()
        self.keep_res = keep_res
        if keep_res is False:
            self.down_sample_block = DownSampleBlock(in_channels, out_channels, instance_norm)
            self.res_blocks = nn.ModuleList([ResBlock(out_channels, out_channels, instance_norm)] * repetitions)
        else:
            self.res_blocks = nn.ModuleList([conv3x3(in_channels, out_channels)] +
                                            [ResBlock(out_channels, out_channels, instance_norm)] * repetitions)

    def forward(self, x):
        if self.keep_res is False:
            net = self.down_sample_block(x)
        else:
            net = x
        for res_block in self.res_blocks:
            net = res_block(net)
        return net


class InvertedSphereFaceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repetitions):
        super(InvertedSphereFaceBlock, self).__init__()
        self.up_sample_block = UpSampleBlock()
        self.res_blocks = nn.ModuleList([conv3x3(in_channels, out_channels)] +
                                        [ResBlock(out_channels, out_channels)] * repetitions)

    def forward(self, x):
        net = self.up_sample_block(x)
        for res_block in self.res_blocks:
            net = res_block(net)
        return net


class StereoSphereRes(nn.Module):
    def __init__(self, input_size=512, input_channels=3, sphereface_size=12, net_dropout_prob=0.1, train_ae=False,
                 multi_inputs_depth=0, bm=False):
        super(StereoSphereRes, self).__init__()

        THIS_FILE = os.path.basename(os.path.realpath(__file__)).split('.')[0] + '.' #RanS 11.1.22
        self.model_name = THIS_FILE + 'StereoSphereRes()' #RanS 11.1.22
        self.bm = bm
        self.input_size = input_size
        self.train_ae = train_ae
        self.multi_inputs_depth = multi_inputs_depth

        res_blocks = cfg[sphereface_size]

        self.block1 = SphereFaceBlock(input_channels, block2channels[0], res_blocks[0], keep_res=True)
        self.block2 = SphereFaceBlock(block2channels[0], block2channels[1], res_blocks[1])
        self.block3 = SphereFaceBlock(block2channels[1], block2channels[2], res_blocks[2])
        self.block4 = SphereFaceBlock(block2channels[2], block2channels[3], res_blocks[3])
        self.block5 = SphereFaceBlock(block2channels[3], block2channels[4], res_blocks[4])

        self.sphereface_blocks = nn.ModuleList([self.block1, self.block2, self.block3, self.block4, self.block5])
        if multi_inputs_depth > 0:

            self.sphereface_blocks = self.sphereface_blocks[multi_inputs_depth:]
            self.block1 = SphereFaceBlock(input_channels, block2channels[0], res_blocks[0], keep_res=True,instance_norm=True)
            self.block1_1 = SphereFaceBlock(input_channels, block2channels[0], res_blocks[0], keep_res=True,instance_norm=True)
            self.block1_2 = SphereFaceBlock(input_channels, block2channels[0], res_blocks[0], keep_res=True, instance_norm=True)
            data2_list = [self.block1]
            data1_list = [self.block1_1]
            data_tot_list = [self.block1_2]
            if multi_inputs_depth > 1:
                self.block2 = SphereFaceBlock(block2channels[0], block2channels[1], res_blocks[1], instance_norm=True)
                self.block2_1 = SphereFaceBlock(block2channels[0], block2channels[1], res_blocks[1], instance_norm=True)
                self.block2_2 = SphereFaceBlock(block2channels[0], block2channels[1], res_blocks[1], instance_norm=True)
                data2_list.append(self.block2)
                data1_list.append(self.block2_1)
                data_tot_list.append(self.block2_2)
            if multi_inputs_depth > 2:
                self.block3 = SphereFaceBlock(block2channels[1], block2channels[2], res_blocks[2], instance_norm=True)
                self.block3_1 = SphereFaceBlock(block2channels[1], block2channels[2], res_blocks[2], instance_norm=True)
                self.block3_2 = SphereFaceBlock(block2channels[1], block2channels[2], res_blocks[2], instance_norm=True)
                data2_list.append(self.block3)
                data1_list.append(self.block3_1)
                data_tot_list.append(self.block3_2)
            self.data2_blocks = nn.ModuleList(data2_list)
            self.data1_blocks = nn.ModuleList(data1_list)
            self.data_tot_blocks = nn.ModuleList(data_tot_list)

        if self.train_ae:
            self.block6 = InvertedSphereFaceBlock(block2channels[4], block2channels[3], res_blocks[4])
            self.block7 = InvertedSphereFaceBlock(block2channels[3], block2channels[2], res_blocks[3])
            self.block8 = InvertedSphereFaceBlock(block2channels[2], block2channels[1], res_blocks[2])
            self.block9 = InvertedSphereFaceBlock(block2channels[1], block2channels[0], res_blocks[1])
            self.block10 = SphereFaceBlock(block2channels[0], input_channels, res_blocks[0], keep_res=True)
            self.decoder_blocks = nn.ModuleList([self.block6, self.block7, self.block8, self.block9, self.block10])
        else:
            f_size = input_size // (2 ** 4)
            self._gap = nn.AvgPool2d((f_size, f_size), stride=1)
            self._final_1x1_conv = nn.Conv2d(in_channels=block2channels[4], out_channels=2, kernel_size=1)
            # self._final_1x1_conv = nn.Linear(in_features=block2channels[4], out_features=2)
            # self._flatten = nn.Flatten()

        self.net_dropout = torch.nn.Dropout(p=net_dropout_prob)

        # nn.init.xavier_uniform_(self.fc_embedding.weight)

    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_im_size(self):
        return self.input_size

    def forward(self, x, image_names=None):
        if self.multi_inputs_depth > 0:
            indices = [i for i in range(len(x))]
            data2_ind = tensor_to_gpu(torch.tensor([i for i in indices if '02-008' in image_names[i]]), True)
            data1_ind = tensor_to_gpu(torch.tensor([i for i in indices if '01-011' in image_names[i]]), True)
            data_tot_ind = tensor_to_gpu(torch.tensor([i for i in indices if '02-008' not in image_names[i] and '01-011'
                                                       not in image_names[i]]), True)

            x2_output = []
            x1_output = []
            tot_output = []

            if len(data2_ind) > 0:
                x_2 = torch.index_select(x, dim=0, index=data2_ind)
                for block in self.data2_blocks:
                    x_2 = block(x_2)
            if len(data1_ind) > 0:
                x_1 = torch.index_select(x, dim=0, index=data1_ind)
                for block in self.data1_blocks:
                    x_1 = block(x_1)
            if len(data_tot_ind) > 0:
                x_total = torch.index_select(x, dim=0, index=data_tot_ind)
                for block in self.data_tot_blocks:
                    x_total = block(x_total)

            outputs = [None] * len(indices)
            for i,global_i in enumerate(data2_ind):
                outputs[global_i] = torch.unsqueeze(x_2[i], dim=0)
            for i,global_i in enumerate(data1_ind):
                outputs[global_i] = torch.unsqueeze(x_1[i], dim=0)
            for i,global_i in enumerate(data_tot_ind):
                outputs[global_i] = torch.unsqueeze(x_total[i], dim=0)

            flag = 0
            x = torch.cat(outputs, 0)
            flag = 0

        # encode
        for block in self.sphereface_blocks:
            x = block(x)

        if self.train_ae:  # decode
            for block in self.decoder_blocks:
                x = block(x)
        else:  # predict class
            x = self.net_dropout(x)
            x = self._gap(x)
            if self.bm:
                features = torch.clone(x)
            x = self._final_1x1_conv(x)

        x = x.squeeze(dim=3) #RanS 11.1.22
        x = x.squeeze(dim=2) #RanS 11.1.22
        if self.bm:
            return x, features
        else:
            return x