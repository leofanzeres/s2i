from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.modules as nnmodules
from torch.nn.modules.module import _addindent
import torchvision.transforms as transforms
import datasets.vegas_visual as vegas_visual
import numpy as np
import sys
import argparse
import utils as ut  # @UnresolvedImport
import values as v


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # @UndefinedVariable
input_height = 96
input_width = 96
input_channels_num = 3
conv1_num_maps = 96
conv2_num_maps = 192
conv3_num_maps = 192
conv4_num_maps = 208
conv5_num_maps = 224
conv6_num_maps = 240
conv7_num_maps = 256
conv8_num_maps = 256
conv9_num_maps = 256
conv10_num_maps = 256
conv11_num_maps = 256
conv12_num_maps = 256
conv12_output_shape = (2,2)
conv13_num_maps = 256
conv13_output_shape = (1,1)
last_conv_flat_dim = conv13_num_maps * conv13_output_shape[0] * conv13_output_shape[1]
fc1_dim = 256
fc2_dim = fc1_dim
fc3_dim = fc1_dim
bias_conv = False
num_classes = 1


class Net(nn.Module):

    def __init__(self, conditioned=False, activation = 'relu', activation_alpha=1.0):
        super(Net, self).__init__()
        self.conditioned = conditioned
        self.activation = activation # activation options: sigmoid, relu, l_relu, elu, celu, selu, tanh
        self.activation_alpha = activation_alpha

        self.conv1 = nn.Conv2d(input_channels_num, conv1_num_maps, kernel_size=(5,5), stride=(2), bias=bias_conv) # original kernel_size=(7,5)
        self.conv1_bn = nn.BatchNorm2d(conv1_num_maps)
        self.conv1_drop = nn.Dropout2d(0.5)

        self.conv2 = nn.Conv2d(conv1_num_maps, conv2_num_maps, kernel_size=(3,3), stride=(2), bias=bias_conv)
        self.conv2_bn = nn.BatchNorm2d(conv2_num_maps)
        self.conv2_drop = nn.Dropout2d(0.5)

        self.conv3 = nn.Conv2d(conv2_num_maps, conv3_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv3_bn = nn.BatchNorm2d(conv3_num_maps)
        self.conv3_drop = nn.Dropout2d(0.5)

        self.conv4 = nn.Conv2d(conv3_num_maps, conv4_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv4_bn = nn.BatchNorm2d(conv4_num_maps)
        self.conv4_drop = nn.Dropout2d(0.5)

        self.conv5 = nn.Conv2d(conv4_num_maps, conv5_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv5_bn = nn.BatchNorm2d(conv5_num_maps)
        self.conv5_drop = nn.Dropout2d(0.5)

        self.conv6 = nn.Conv2d(conv5_num_maps, conv6_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv6_bn = nn.BatchNorm2d(conv6_num_maps)
        self.conv6_drop = nn.Dropout2d(0.5)

        self.conv7 = nn.Conv2d(conv6_num_maps, conv7_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv7_bn = nn.BatchNorm2d(conv7_num_maps)
        self.conv7_drop = nn.Dropout2d(0.5)

        self.conv8 = nn.Conv2d(conv7_num_maps, conv8_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv8_bn = nn.BatchNorm2d(conv8_num_maps)
        self.conv8_drop = nn.Dropout2d(0.5)

        self.conv9 = nn.Conv2d(conv8_num_maps, conv9_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv9_bn = nn.BatchNorm2d(conv9_num_maps)
        self.conv9_drop = nn.Dropout2d(0.5)

        self.conv10 = nn.Conv2d(conv9_num_maps, conv10_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv10_bn = nn.BatchNorm2d(conv10_num_maps)
        self.conv10_drop = nn.Dropout2d(0.5)

        self.conv11 = nn.Conv2d(conv10_num_maps, conv11_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv11_bn = nn.BatchNorm2d(conv11_num_maps)
        self.conv11_drop = nn.Dropout2d(0.5)

        self.conv12 = nn.Conv2d(conv11_num_maps, conv12_num_maps, kernel_size=(3,3), stride=(1), bias=bias_conv)
        self.conv12_bn = nn.BatchNorm2d(conv12_num_maps)
        self.conv12_drop = nn.Dropout2d(0.5)

        if self.conditioned:
            input_dim = conv12_num_maps + int(v.AUDIO_EMBEDDING_DIMENSION/(conv12_output_shape[0]*conv12_output_shape[1]))
        else:
            input_dim = conv12_num_maps
        self.conv13 = nn.Conv2d(input_dim, num_classes, kernel_size=(2,2), stride=(1), bias=bias_conv)
        self.conv13_bn = nn.BatchNorm2d(num_classes)
        self.conv13_drop = nn.Dropout2d(0.5)


    def forward(self, x, emb=None):

        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv1_drop(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv2_drop(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv3_drop(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv4_drop(x)

        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv5_drop(x)

        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv6_drop(x)

        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv7_drop(x)

        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv8_drop(x)

        x = self.conv9(x)
        x = self.conv9_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv9_drop(x)

        x = self.conv10(x)
        x = self.conv10_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv10_drop(x)

        x = self.conv11(x)
        x = self.conv11_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv11_drop(x)

        x = self.conv12(x)
        x = self.conv12_bn(x)
        x = ut.get_activated(x, self.activation, alph=self.activation_alpha)
        x = self.conv12_drop(x)

        if emb is not None:
            emb_shape = emb.size()
            emb = emb.view(emb_shape[0], -1, conv12_output_shape[0], conv12_output_shape[1])
            x = torch.cat([x, emb], 1)  # @UndefinedVariable

        x = self.conv13(x)
        x = self.conv13_bn(x)
        x = ut.get_activated(x, 'tanh')

        return x


    def summary(self):
        for param in self.parameters():
            print(param.size())


    def torch_summarize(self, show_weights=True, show_parameters=True):
        """Summarizes torch model by showing trainable parameters and weights."""
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                nnmodules.container.Container,
                nnmodules.container.Sequential
            ]:
                modstr = self.torch_summarize(module)
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr +=  ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'

        return tmpstr


def get_train_loader():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = vegas_visual.VEGAS_VISUAL(root=v.BATCHES_VISUAL_DIR,
                                         input_height=input_height, input_width=input_width, input_channels_num=input_channels_num,
                                         train=True, transform=transform)  # @UndefinedVariable

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)  # @UndefinedVariable

    return trainloader


def get_test_loader():

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = vegas_visual.VEGAS_VISUAL(root=v.BATCHES_VISUAL_DIR,
                                        input_height=input_height, input_width=input_width, input_channels_num=input_channels_num,
                                        train=False, transform=transform_test)  # @UndefinedVariable

    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)  # @UndefinedVariable

    return testloader
