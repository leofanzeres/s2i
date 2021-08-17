from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.modules as nnmodules
from torch.nn.modules.module import _addindent
import torchvision.transforms as transforms
import datasets.vegas_visual_generator as vegas_visual_generator
import numpy as np
import argparse
import utils as ut  # @UnresolvedImport
import values as v


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # @UndefinedVariable
audio_embedding_dim = v.AUDIO_EMBEDDING_DIMENSION
input_height = 128
input_width = 100
target_height = 96
target_width = 96
target_channels_num = 3
label_length = 11
input_embedding_dim = v.AUDIO_EMBEDDING_DIMENSION
dropout = (1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0)


class Net(nn.Module):

    def __init__(self, num_classes, embedding_dimension, activation = 'relu', activation_alpha=1.0):
        super(Net, self).__init__()

        # Get dimensions
        (input_feature_map_shape,input_channels_num,convT1_num_maps,convT2_num_maps,convT3_num_maps,convT4_num_maps,convT5_num_maps,convT6_num_maps,
            convT7_num_maps,convT8_num_maps,convT9_num_maps,convT10_num_maps,convT11_num_maps,convT12_num_maps,convT13_num_maps,convT14_num_maps,
            convT15_num_maps,convT16_num_maps,convT17_num_maps,convT18_num_maps) = get_dimensions(embedding_dimension)

        self.input_feature_map_shape = input_feature_map_shape
        self.input_channels_num = input_channels_num
        self.activation = [activation]*24 # activation options: sigmoid, relu, l_relu, elu, celu, selu, tanh
        self.activation_alpha = [activation_alpha]*24

        # Transition 0
        self.transition0 = nn.ConvTranspose2d(in_channels = input_channels_num, out_channels = convT1_num_maps, kernel_size = (2,2), stride = 1, padding=0)
        self.transition0_bn = nn.BatchNorm2d(convT1_num_maps)
        if dropout[0]>0: self.transition0_drop = nn.Dropout2d(0.15)

        # Dense Block 1

        self.convT1 = nn.ConvTranspose2d(in_channels = convT1_num_maps, out_channels = convT1_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT1_bn = nn.BatchNorm2d(convT1_num_maps)
        if dropout[1]>0: self.convT1_drop = nn.Dropout2d(0.0)

        self.convT2 = nn.ConvTranspose2d(in_channels = convT1_num_maps, out_channels = convT2_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT2_bn = nn.BatchNorm2d(convT2_num_maps)
        if dropout[2]>0: self.convT2_drop = nn.Dropout2d(0.0)

        self.convT3 = nn.ConvTranspose2d(in_channels = convT1_num_maps+convT2_num_maps, out_channels = convT3_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT3_bn = nn.BatchNorm2d(convT3_num_maps)
        if dropout[3]>0: self.convT3_drop = nn.Dropout2d(0.0)

        # Transition 1
        self.transition1 = nn.ConvTranspose2d(in_channels = convT1_num_maps+convT2_num_maps+convT3_num_maps, out_channels = convT3_num_maps, kernel_size = (3,3), stride = 1, padding=0)
        self.transition1_bn = nn.BatchNorm2d(convT3_num_maps)
        if dropout[4]>0: self.transition1_drop = nn.Dropout2d(0.13)

        # Dense Block 2
        self.convT4 = nn.ConvTranspose2d(in_channels = convT3_num_maps, out_channels = convT4_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT4_bn = nn.BatchNorm2d(convT4_num_maps)
        if dropout[5]>0: self.convT4_drop = nn.Dropout2d(0.0)

        self.convT5 = nn.ConvTranspose2d(in_channels = convT4_num_maps, out_channels = convT5_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT5_bn = nn.BatchNorm2d(convT5_num_maps)
        if dropout[6]>0: self.convT5_drop = nn.Dropout2d(0.0)

        self.convT6 = nn.ConvTranspose2d(in_channels = convT4_num_maps+convT5_num_maps, out_channels = convT6_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT6_bn = nn.BatchNorm2d(convT6_num_maps)
        if dropout[7]>0: self.convT6_drop = nn.Dropout2d(0.0)

        # Transition 2
        self.transition2 = nn.ConvTranspose2d(in_channels = convT4_num_maps+convT5_num_maps+convT6_num_maps, out_channels = convT6_num_maps, kernel_size = (3,3), stride = 1, padding=0)
        self.transition2_bn = nn.BatchNorm2d(convT6_num_maps)
        if dropout[8]>0: self.transition2_drop = nn.Dropout2d(0.11)

        # Dense Block 3

        self.convT7 = nn.ConvTranspose2d(in_channels = convT6_num_maps, out_channels = convT7_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT7_bn = nn.BatchNorm2d(convT7_num_maps)
        if dropout[9]>0: self.convT7_drop = nn.Dropout2d(0.0)

        self.convT8 = nn.ConvTranspose2d(in_channels = convT7_num_maps, out_channels = convT8_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT8_bn = nn.BatchNorm2d(convT8_num_maps)
        if dropout[10]>0: self.convT8_drop = nn.Dropout2d(0.0)

        self.convT9 = nn.ConvTranspose2d(in_channels = convT7_num_maps+convT8_num_maps, out_channels = convT9_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT9_bn = nn.BatchNorm2d(convT9_num_maps)
        if dropout[11]>0: self.convT9_drop = nn.Dropout2d(0.0)

        # Transition 3
        self.transition3 = nn.ConvTranspose2d(in_channels = convT7_num_maps+convT8_num_maps+convT9_num_maps, out_channels = convT9_num_maps, kernel_size = (4,4), stride = 2, padding=1)
        self.transition3_bn = nn.BatchNorm2d(convT9_num_maps)
        if dropout[12]>0: self.transition3_drop = nn.Dropout2d(0.09)

        # Dense Block 4

        self.convT10 = nn.ConvTranspose2d(in_channels = convT9_num_maps, out_channels = convT10_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT10_bn = nn.BatchNorm2d(convT10_num_maps)
        if dropout[13]>0: self.convT10_drop = nn.Dropout2d(0.0)

        self.convT11 = nn.ConvTranspose2d(in_channels = convT10_num_maps, out_channels = convT11_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT11_bn = nn.BatchNorm2d(convT11_num_maps)
        if dropout[14]>0: self.convT11_drop = nn.Dropout2d(0.0)

        self.convT12 = nn.ConvTranspose2d(in_channels = convT10_num_maps+convT11_num_maps, out_channels = convT12_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT12_bn = nn.BatchNorm2d(convT12_num_maps)
        if dropout[15]>0: self.convT12_drop = nn.Dropout2d(0.0)

        # Transition 4
        self.transition4 = nn.ConvTranspose2d(in_channels = convT10_num_maps+convT11_num_maps+convT12_num_maps, out_channels = convT12_num_maps, kernel_size = (4,4), stride = 2, padding=1)
        self.transition4_bn = nn.BatchNorm2d(convT12_num_maps)
        if dropout[16]>0: self.transition4_drop = nn.Dropout2d(0.07)

        # Dense Block 5

        self.convT13 = nn.ConvTranspose2d(in_channels = convT12_num_maps, out_channels = convT13_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT13_bn = nn.BatchNorm2d(convT13_num_maps)
        if dropout[17]>0: self.convT13_drop = nn.Dropout2d(0.0)

        self.convT14 = nn.ConvTranspose2d(in_channels = convT13_num_maps, out_channels = convT14_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT14_bn = nn.BatchNorm2d(convT14_num_maps)
        if dropout[18]>0: self.convT14_drop = nn.Dropout2d(0.0)

        self.convT15 = nn.ConvTranspose2d(in_channels = convT13_num_maps+convT14_num_maps, out_channels = convT15_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT15_bn = nn.BatchNorm2d(convT15_num_maps)
        if dropout[19]>0: self.convT15_drop = nn.Dropout2d(0.0)

        # Transition 5
        self.transition5 = nn.ConvTranspose2d(in_channels = convT13_num_maps+convT14_num_maps+convT15_num_maps, out_channels = convT15_num_maps, kernel_size = (4,4), stride = 2, padding=1)
        self.transition5_bn = nn.BatchNorm2d(convT15_num_maps)
        if dropout[20]>0: self.transition5_drop = nn.Dropout2d(0.05)

        # Dense Block 6

        self.convT16 = nn.ConvTranspose2d(in_channels = convT15_num_maps, out_channels = convT16_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT16_bn = nn.BatchNorm2d(convT16_num_maps)
        if dropout[21]>0: self.convT16_drop = nn.Dropout2d(0.0)

        self.convT17 = nn.ConvTranspose2d(in_channels = convT16_num_maps, out_channels = convT17_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT17_bn = nn.BatchNorm2d(convT17_num_maps)
        if dropout[22]>0: self.convT17_drop = nn.Dropout2d(0.0)

        self.convT18 = nn.ConvTranspose2d(in_channels = convT16_num_maps+convT17_num_maps, out_channels = convT18_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        self.convT18_bn = nn.BatchNorm2d(convT18_num_maps)
        if dropout[23]>0: self.convT18_drop = nn.Dropout2d(0.0)

        # Transition 6
        self.transition6 = nn.ConvTranspose2d(in_channels = convT16_num_maps+convT17_num_maps+convT18_num_maps, out_channels = target_channels_num, kernel_size = (4,4), stride = 2, padding=1)
        self.transition6_bn = nn.BatchNorm2d(target_channels_num)


    def forward(self, x):

        x = x.view(-1, self.input_channels_num, self.input_feature_map_shape[0], self.input_feature_map_shape[1])

        x = self.transition0(x)
        x = self.transition0_bn(x)
        x = ut.get_activated(x, self.activation[0], alph=self.activation_alpha[0])
        if dropout[0]>0: x = self.transition0_drop(x)

        x1 = self.convT1(x)
        x1 = self.convT1_bn(x1)
        x1 = ut.get_activated(x1, self.activation[1], alph=self.activation_alpha[1])
        if dropout[1]>0: x1 = self.convT1_drop(x1)

        x2 = self.convT2(x1)
        x2 = self.convT2_bn(x2)
        x2 = ut.get_activated(x2, self.activation[2], alph=self.activation_alpha[2])
        if dropout[2]>0: x2 = self.convT2_drop(x2)
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable

        x3 = self.convT3(x2_dense)
        x3 = self.convT3_bn(x3)
        x3 = ut.get_activated(x3, self.activation[3], alph=self.activation_alpha[3])
        if dropout[3]>0: x3 = self.convT3_drop(x3)
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable

        x = self.transition1(x3_dense)
        x = self.transition1_bn(x)
        x = ut.get_activated(x, self.activation[4], alph=self.activation_alpha[4])
        if dropout[4]>0: x = self.transition1_drop(x)

        x1 = self.convT4(x)
        x1 = self.convT4_bn(x1)
        x1 = ut.get_activated(x1, self.activation[5], alph=self.activation_alpha[5])
        if dropout[5]>0: x1 = self.convT4_drop(x1)

        x2 = self.convT5(x1)
        x2 = self.convT5_bn(x2)
        x2 = ut.get_activated(x2, self.activation[6], alph=self.activation_alpha[6])
        if dropout[6]>0: x2 = self.convT5_drop(x2)
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable

        x3 = self.convT6(x2_dense)
        x3 = self.convT6_bn(x3)
        x3 = ut.get_activated(x3, self.activation[7], alph=self.activation_alpha[7])
        if dropout[7]>0: x3 = self.convT6_drop(x3)
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable

        x = self.transition2(x3_dense)
        x = self.transition2_bn(x)
        x = ut.get_activated(x, self.activation[8], alph=self.activation_alpha[8])
        if dropout[8]>0: x = self.transition2_drop(x)

        x1 = self.convT7(x)
        x1 = self.convT7_bn(x1)
        x1 = ut.get_activated(x1, self.activation[9], alph=self.activation_alpha[9])
        if dropout[9]>0: x1 = self.convT7_drop(x1)

        x2 = self.convT8(x1)
        x2 = self.convT8_bn(x2)
        x2 = ut.get_activated(x2, self.activation[10], alph=self.activation_alpha[10])
        if dropout[10]>0: x2 = self.convT8_drop(x2)
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable

        x3 = self.convT9(x2_dense)
        x3 = self.convT9_bn(x3)
        x3 = ut.get_activated(x3, self.activation[11], alph=self.activation_alpha[11])
        if dropout[11]>0: x3 = self.convT9_drop(x3)
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable

        x = self.transition3(x3_dense)
        x = self.transition3_bn(x)
        x = ut.get_activated(x, self.activation[12], alph=self.activation_alpha[12])
        if dropout[12]>0: x = self.transition3_drop(x)

        x1 = self.convT10(x)
        x1 = self.convT10_bn(x1)
        x1 = ut.get_activated(x1, self.activation[13], alph=self.activation_alpha[13])
        if dropout[13]>0: x1 = self.convT10_drop(x1)

        x2 = self.convT11(x1)
        x2 = self.convT11_bn(x2)
        x2 = ut.get_activated(x2, self.activation[14], alph=self.activation_alpha[14])
        if dropout[14]>0: x2 = self.convT11_drop(x2)
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable

        x3 = self.convT12(x2_dense)
        x3 = self.convT12_bn(x3)
        x3 = ut.get_activated(x3, self.activation[15], alph=self.activation_alpha[15])
        if dropout[15]>0: x3 = self.convT12_drop(x3)
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable

        x = self.transition4(x3_dense)
        x = self.transition4_bn(x)
        x = ut.get_activated(x, self.activation[16], alph=self.activation_alpha[16])
        if dropout[16]>0: x = self.transition4_drop(x)

        x1 = self.convT13(x)
        x1 = self.convT13_bn(x1)
        x1 = ut.get_activated(x1, self.activation[17], alph=self.activation_alpha[17])
        if dropout[17]>0: x1 = self.convT13_drop(x1)

        x2 = self.convT14(x1)
        x2 = self.convT14_bn(x2)
        x2 = ut.get_activated(x2, self.activation[18], alph=self.activation_alpha[18])
        if dropout[18]>0: x2 = self.convT14_drop(x2)
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable

        x3 = self.convT15(x2_dense)
        x3 = self.convT15_bn(x3)
        x3 = ut.get_activated(x3, self.activation[19], alph=self.activation_alpha[19])
        if dropout[19]>0: x3 = self.convT15_drop(x3)
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable

        x = self.transition5(x3_dense)
        x = self.transition5_bn(x)
        x = ut.get_activated(x, self.activation[20], alph=self.activation_alpha[20])
        if dropout[20]>0: x = self.transition5_drop(x)

        x1 = self.convT16(x)
        x1 = self.convT16_bn(x1)
        x1 = ut.get_activated(x1, self.activation[21], alph=self.activation_alpha[21])
        if dropout[21]>0: x1 = self.convT16_drop(x1)

        x2 = self.convT17(x1)
        x2 = self.convT17_bn(x2)
        x2 = ut.get_activated(x2, self.activation[22], alph=self.activation_alpha[22])
        if dropout[22]>0: x2 = self.convT17_drop(x2)
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable

        x3 = self.convT18(x2_dense)
        x3 = self.convT18_bn(x3)
        x3 = ut.get_activated(x3, self.activation[23], alph=self.activation_alpha[23])
        if dropout[23]>0: x3 = self.convT18_drop(x3)
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable

        x = self.transition6(x3_dense)

        x = self.transition6_bn(x)

        x = torch.tanh(x)  # @UndefinedVariable

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


def get_train_loader(shuffle=True):

    transform = None # No transformation because it is the embedding that is being loaded

    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = vegas_visual_generator.VEGAS_VISUAL_GENERATOR(
            root=v.BATCHES_AUDIO_VISUAL_EMBEDDING_TO_IMAGE_DIR, label_length=label_length, input_length=audio_embedding_dim, target_height=target_height,
            target_width=target_width, target_channels_num=target_channels_num, train=True, transform=transform, target_transform=target_transform)  # @UndefinedVariable

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=shuffle, num_workers=2)  # @UndefinedVariable

    return trainloader


def get_test_loader(shuffle=False):

    transform_test = None # No transformation because it is the embedding that is being loaded

    target_transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = vegas_visual_generator.VEGAS_VISUAL_GENERATOR(
            root=v.BATCHES_AUDIO_VISUAL_EMBEDDING_TO_IMAGE_DIR, label_length=label_length, input_length=audio_embedding_dim, target_height=target_height,
            target_width=target_width, target_channels_num=target_channels_num, train=False, transform=transform_test, target_transform=target_transform_test)  # @UndefinedVariable

    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=shuffle, num_workers=2)  # @UndefinedVariable

    return testloader


def get_dimensions (input_embedding_dim):
    input_feature_map_shape = (1,1)
    input_channels_num = input_embedding_dim
    convT1_num_maps = 512
    convT2_num_maps = 512
    convT3_num_maps = 512
    convT4_num_maps = 512
    convT5_num_maps = 416
    convT6_num_maps = 352
    convT7_num_maps = 304
    convT8_num_maps = 256
    convT9_num_maps = 224
    convT10_num_maps = 192
    convT11_num_maps = 176
    convT12_num_maps = 160
    convT13_num_maps = 144
    convT14_num_maps = 128
    convT15_num_maps = 112
    convT16_num_maps = 96
    convT17_num_maps = 80
    convT18_num_maps = 64

    return (input_feature_map_shape,input_channels_num,convT1_num_maps,convT2_num_maps,convT3_num_maps,convT4_num_maps,convT5_num_maps,convT6_num_maps,
            convT7_num_maps,convT8_num_maps,convT9_num_maps,convT10_num_maps,convT11_num_maps,convT12_num_maps,convT13_num_maps,convT14_num_maps,
            convT15_num_maps,convT16_num_maps,convT17_num_maps,convT18_num_maps)
