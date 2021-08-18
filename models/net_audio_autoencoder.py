from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.modules as nnmodules
from torch.nn.modules.module import _addindent
import torchvision.transforms as transforms
import datasets.vegas_audio as vegas_audio
import numpy as np
import utils as ut
import values as v


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # @UndefinedVariable
input_height = 128
input_width = 100
target_height = input_height
target_width = input_width
input_channels_num = 1 # 128x100=12800
conv_num_maps = (80,176,192,224,272,304,352,416,512,512,512,512,v.AUDIO_EMBEDDING_DIMENSION)
last_feature_map_shape = (1,1)
last_conv_flat_dim = conv_num_maps[12]*last_feature_map_shape[0]*last_feature_map_shape[1]


class Net(nn.Module):
    last_conv_flat_dim
    def __init__(self, num_classes, mode='auto', activation='relu', activation_alpha=1.0):
        super(Net, self).__init__()
        self.mode = mode
        self.activation = [activation]*24 # activation options: sigmoid, relu, l_relu, softplus, elu, celu, selu, tanh
        self.activation_alpha = [activation_alpha]*24

        # ENCODER

        self.conv1 = nn.Conv2d(input_channels_num, conv_num_maps[0], kernel_size=(7,5), stride=(2))
        self.conv1_bn = nn.BatchNorm2d(conv_num_maps[0])

        self.conv2 = nn.Conv2d(conv_num_maps[0], conv_num_maps[1], kernel_size=(5,3), stride=(2))
        self.conv2_bn = nn.BatchNorm2d(conv_num_maps[1])

        self.conv3 = nn.Conv2d(conv_num_maps[1], conv_num_maps[2], kernel_size=(5,3), stride=(1))
        self.conv3_bn = nn.BatchNorm2d(conv_num_maps[2])

        self.conv4 = nn.Conv2d(conv_num_maps[2], conv_num_maps[3], kernel_size=(5,3), stride=(1))
        self.conv4_bn = nn.BatchNorm2d(conv_num_maps[3])

        self.conv5 = nn.Conv2d(conv_num_maps[3], conv_num_maps[4], kernel_size=(5,3), stride=(1))
        self.conv5_bn = nn.BatchNorm2d(conv_num_maps[4])

        self.conv6 = nn.Conv2d(conv_num_maps[4], conv_num_maps[5], kernel_size=(3,3), stride=(1))
        self.conv6_bn = nn.BatchNorm2d(conv_num_maps[5])

        self.conv7 = nn.Conv2d(conv_num_maps[5], conv_num_maps[6], kernel_size=(3,3), stride=(1))
        self.conv7_bn = nn.BatchNorm2d(conv_num_maps[6])

        self.conv8 = nn.Conv2d(conv_num_maps[6], conv_num_maps[7], kernel_size=(3,3), stride=(1))
        self.conv8_bn = nn.BatchNorm2d(conv_num_maps[7])

        self.conv9 = nn.Conv2d(conv_num_maps[7], conv_num_maps[8], kernel_size=(3,3), stride=(1))
        self.conv9_bn = nn.BatchNorm2d(conv_num_maps[8])

        self.conv10 = nn.Conv2d(conv_num_maps[8], conv_num_maps[9], kernel_size=(3,3), stride=(1))
        self.conv10_bn = nn.BatchNorm2d(conv_num_maps[9])

        self.conv11 = nn.Conv2d(conv_num_maps[9], conv_num_maps[10], kernel_size=(3,3), stride=(1))
        self.conv11_bn = nn.BatchNorm2d(conv_num_maps[10])

        self.conv12 = nn.Conv2d(conv_num_maps[10], conv_num_maps[11], kernel_size=(3,3), stride=(1))
        self.conv12_bn = nn.BatchNorm2d(conv_num_maps[11])

        self.conv13 = nn.Conv2d(conv_num_maps[11], conv_num_maps[12], kernel_size=(3,3), stride=(1))
        self.conv13_bn = nn.BatchNorm2d(conv_num_maps[12])

        # DECODER

        self.convT1 = nn.ConvTranspose2d(in_channels = conv_num_maps[12], out_channels = conv_num_maps[11], kernel_size = (3,3), stride = 1, padding=0)
        self.convT1_bn = nn.BatchNorm2d(conv_num_maps[11])
        self.convT1_drop = nn.Dropout2d(0.15)

        self.convT2 = nn.ConvTranspose2d(in_channels = conv_num_maps[11], out_channels = conv_num_maps[10], kernel_size = (3,3), stride = 1, padding=0)
        self.convT2_bn = nn.BatchNorm2d(conv_num_maps[10])
        self.convT2_drop = nn.Dropout2d(0.14)

        self.convT3 = nn.ConvTranspose2d(in_channels = conv_num_maps[10], out_channels = conv_num_maps[9], kernel_size = (3,3), stride = 1, padding=0)
        self.convT3_bn = nn.BatchNorm2d(conv_num_maps[9])
        self.convT3_drop = nn.Dropout2d(0.13)

        self.convT4 = nn.ConvTranspose2d(in_channels = conv_num_maps[9], out_channels = conv_num_maps[8], kernel_size = (3,3), stride = 1, padding=0)
        self.convT4_bn = nn.BatchNorm2d(conv_num_maps[8])
        self.convT4_drop = nn.Dropout2d(0.12)

        self.convT5 = nn.ConvTranspose2d(in_channels = conv_num_maps[8], out_channels = conv_num_maps[7], kernel_size = (3,3), stride = 1, padding=0)
        self.convT5_bn = nn.BatchNorm2d(conv_num_maps[7])
        self.convT5_drop = nn.Dropout2d(0.11)

        self.convT6 = nn.ConvTranspose2d(in_channels = conv_num_maps[7], out_channels = conv_num_maps[6], kernel_size = (3,3), stride = 1, padding=0)
        self.convT6_bn = nn.BatchNorm2d(conv_num_maps[6])
        self.convT6_drop = nn.Dropout2d(0.1)

        self.convT7 = nn.ConvTranspose2d(in_channels = conv_num_maps[6], out_channels = conv_num_maps[5], kernel_size = (3,3), stride = 1, padding=0)
        self.convT7_bn = nn.BatchNorm2d(conv_num_maps[5])
        self.convT7_drop = nn.Dropout2d(0.1)

        self.convT8 = nn.ConvTranspose2d(in_channels = conv_num_maps[5], out_channels = conv_num_maps[4], kernel_size = (3,3), stride = 1, padding=0)
        self.convT8_bn = nn.BatchNorm2d(conv_num_maps[4])
        self.convT8_drop = nn.Dropout2d(0.09)

        self.convT9 = nn.ConvTranspose2d(in_channels = conv_num_maps[4], out_channels = conv_num_maps[3], kernel_size = (5,3), stride = 1, padding=0)
        self.convT9_bn = nn.BatchNorm2d(conv_num_maps[3])
        self.convT9_drop = nn.Dropout2d(0.08)

        self.convT10 = nn.ConvTranspose2d(in_channels = conv_num_maps[3], out_channels = conv_num_maps[2], kernel_size = (5,3), stride = 1, padding=0)
        self.convT10_bn = nn.BatchNorm2d(conv_num_maps[2])
        self.convT10_drop = nn.Dropout2d(0.07)

        self.convT11 = nn.ConvTranspose2d(in_channels = conv_num_maps[2], out_channels = conv_num_maps[1], kernel_size = (5,3), stride = 1, padding=0)
        self.convT11_bn = nn.BatchNorm2d(conv_num_maps[1])
        self.convT11_drop = nn.Dropout2d(0.06)

        self.convT12 = nn.ConvTranspose2d(in_channels = conv_num_maps[1], out_channels = conv_num_maps[0], kernel_size = (5,4), stride = 2, padding=0)
        self.convT12_bn = nn.BatchNorm2d(conv_num_maps[0])
        self.convT12_drop = nn.Dropout2d(0.05)

        self.convT13 = nn.ConvTranspose2d(in_channels = conv_num_maps[0], out_channels = input_channels_num, kernel_size = (8,6), stride = 2, padding=0)
        self.convT13_bn = nn.BatchNorm2d(input_channels_num)


    def forward(self, x):

        if (self.mode == 'auto') | (self.mode == 'encode'):

            x = self.conv1(x)
            x = self.conv1_bn(x)
            x = ut.get_activated(x, self.activation[0], alph=self.activation_alpha[0])

            x = self.conv2(x)
            x = self.conv2_bn(x)
            x = ut.get_activated(x, self.activation[1], alph=self.activation_alpha[1])

            x = self.conv3(x)
            x = self.conv3_bn(x)
            x = ut.get_activated(x, self.activation[2], alph=self.activation_alpha[2])

            x = self.conv4(x)
            x = self.conv4_bn(x)
            x = ut.get_activated(x, self.activation[3], alph=self.activation_alpha[3])

            x = self.conv5(x)
            x = self.conv5_bn(x)
            x = ut.get_activated(x, self.activation[4], alph=self.activation_alpha[4])

            x = self.conv6(x)
            x = self.conv6_bn(x)
            x = ut.get_activated(x, self.activation[5], alph=self.activation_alpha[5])

            x = self.conv7(x)
            x = self.conv7_bn(x)
            x = ut.get_activated(x, self.activation[6], alph=self.activation_alpha[6])

            x = self.conv8(x)
            x = self.conv8_bn(x)
            x = ut.get_activated(x, self.activation[7], alph=self.activation_alpha[7])

            x = self.conv9(x)
            x = self.conv9_bn(x)
            x = ut.get_activated(x, self.activation[8], alph=self.activation_alpha[8])

            x = self.conv10(x)
            x = self.conv10_bn(x)
            x = ut.get_activated(x, self.activation[9], alph=self.activation_alpha[9])

            x = self.conv11(x)
            x = self.conv11_bn(x)
            x = ut.get_activated(x, self.activation[10], alph=self.activation_alpha[10])

            x = self.conv12(x)
            x = self.conv12_bn(x)
            x = ut.get_activated(x, self.activation[11], alph=self.activation_alpha[11])

            x = self.conv13(x)
            x = self.conv13_bn(x)
            x = torch.tanh(x)  # @UndefinedVariable

            embeddings = x.view(-1, last_conv_flat_dim)


        if (self.mode == 'auto') | (self.mode == 'decode'):

            if self.mode == 'decode':
                embeddings = x
                x = x.view(-1, conv_num_maps[12], last_feature_map_shape[0], last_feature_map_shape[1])

            x = self.convT1(x)
            x = self.convT1_bn(x)
            x = ut.get_activated(x, self.activation[12], alph=self.activation_alpha[12])
            x = self.convT1_drop(x)

            x = self.convT2(x)
            x = self.convT2_bn(x)
            x = ut.get_activated(x, self.activation[13], alph=self.activation_alpha[13])
            x = self.convT2_drop(x)

            x = self.convT3(x)
            x = self.convT3_bn(x)
            x = ut.get_activated(x, self.activation[14], alph=self.activation_alpha[14])
            x = self.convT3_drop(x)

            x = self.convT4(x)
            x = self.convT4_bn(x)
            x = ut.get_activated(x, self.activation[15], alph=self.activation_alpha[15])
            x = self.convT4_drop(x)

            x = self.convT5(x)
            x = self.convT5_bn(x)
            x = ut.get_activated(x, self.activation[16], alph=self.activation_alpha[16])
            x = self.convT5_drop(x)

            x = self.convT6(x)
            x = self.convT6_bn(x)
            x = ut.get_activated(x, self.activation[17], alph=self.activation_alpha[17])
            x = self.convT6_drop(x)

            x = self.convT7(x)
            x = self.convT7_bn(x)
            x = ut.get_activated(x, self.activation[18], alph=self.activation_alpha[18])
            x = self.convT7_drop(x)

            x = self.convT8(x)
            x = self.convT8_bn(x)
            x = ut.get_activated(x, self.activation[19], alph=self.activation_alpha[19])
            x = self.convT8_drop(x)

            x = self.convT9(x)
            x = self.convT9_bn(x)
            x = ut.get_activated(x, self.activation[20], alph=self.activation_alpha[20])
            x = self.convT9_drop(x)

            x = self.convT10(x)
            x = self.convT10_bn(x)
            x = ut.get_activated(x, self.activation[21], alph=self.activation_alpha[21])
            x = self.convT10_drop(x)

            x = self.convT11(x)
            x = self.convT11_bn(x)
            x = ut.get_activated(x, self.activation[22], alph=self.activation_alpha[22])
            x = self.convT11_drop(x)

            x = self.convT12(x)
            x = self.convT12_bn(x)
            x = ut.get_activated(x, self.activation[23], alph=self.activation_alpha[23])
            x = self.convT12_drop(x)

            x = self.convT13(x)
            x = self.convT13_bn(x)
            x = torch.tanh(x)  # @UndefinedVariable

        return x, embeddings


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
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = vegas_audio.VEGAS_AUDIO(root=v.BATCHES_AUDIO_DIR,
                                       input_height=input_height, input_width=input_width, input_channels_num=input_channels_num,
                                       train=True, transform=transform)  # @UndefinedVariable
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)  # @UndefinedVariable
    return trainloader

def get_test_loader():
    transform_test = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = vegas_audio.VEGAS_AUDIO(root=v.BATCHES_AUDIO_DIR,
                                      input_height=input_height, input_width=input_width, input_channels_num=input_channels_num,
                                      train=False, transform=transform_test)  # @UndefinedVariable

    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)  # @UndefinedVariable
    return testloader
