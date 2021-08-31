from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.modules as nnmodules
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
import utils as ut  # @UnresolvedImport
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # @UndefinedVariable
input_height = 96
input_width = 96
input_channels_num = 3
stride = 2
conv1_num_maps = 12
conv2_num_maps = 32
conv3_num_maps = 32
conv4_num_maps = 32
conv5_num_maps = 32
fc_dim = 32
fc1_pass = False
num_classes = 2
non_interpretable_image_label = 0
activation = 'relu' # options: sigmoid, relu, l_relu, softplus, elu, celu, selu, tanh
activation_alpha = 1.0


class Net(nn.Module):
    
    def __init__(self, input_length):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels_num, conv1_num_maps, kernel_size=(5,5), stride=(stride))
        self.conv1_drop = nn.Dropout2d(0.1)
        self.conv1_bn = nn.BatchNorm2d(conv1_num_maps)
        self.conv2 = nn.Conv2d(conv1_num_maps, conv2_num_maps, kernel_size=(3,3), stride=(stride))
        self.conv2_drop = nn.Dropout2d(0.2)
        self.conv2_bn = nn.BatchNorm2d(conv2_num_maps)
        self.conv3 = nn.Conv2d(conv2_num_maps, conv3_num_maps, kernel_size=(3,3), stride=(stride))
        self.conv3_drop = nn.Dropout2d(0.3)
        self.conv3_bn = nn.BatchNorm2d(conv3_num_maps)
        self.conv4 = nn.Conv2d(conv3_num_maps, conv4_num_maps, kernel_size=(3,3), stride=(stride))
        self.conv4_drop = nn.Dropout2d(0.4)
        self.conv4_bn = nn.BatchNorm2d(conv4_num_maps)
        self.conv5 = nn.Conv2d(conv4_num_maps, conv5_num_maps, kernel_size=(3,3), stride=(stride))
        self.conv5_drop = nn.Dropout2d(0.5)
        self.conv5_bn = nn.BatchNorm2d(conv5_num_maps)
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc1_bn = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc2_bn = nn.BatchNorm1d(fc_dim)
        self.fc3 = nn.Linear(fc_dim, num_classes)
                
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = ut.get_activated(x, activation, alph=activation_alpha)
        x = self.conv1_drop(x)
         
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = ut.get_activated(x, activation, alph=activation_alpha)
        x = self.conv2_drop(x)
         
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = ut.get_activated(x, activation, alph=activation_alpha)
        x = self.conv3_drop(x)
         
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = ut.get_activated(x, activation, alph=activation_alpha)
        x = self.conv4_drop(x)
         
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = ut.get_activated(x, activation, alph=activation_alpha)
        x = self.conv5_drop(x)        
        
        x = x.view(-1, fc_dim)
        
        if fc1_pass:
            x = self.fc1(x)
            x = self.fc1_bn(x)
            x = ut.get_activated(x, activation, alph=activation_alpha)
            x = self.fc1_drop(x)
          
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = ut.get_activated(x, activation, alph=activation_alpha)
        x = self.fc2_drop(x)

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        
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

    


