from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.modules as nnmodules
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
import torchvision.transforms as transforms
import datasets.vegas_audio as vegas_audio
import utils as ut  # @UnresolvedImport
import values as v


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # @UndefinedVariable
input_height = 128
input_width = 100
input_channels_num = 1
conv1_num_maps = 32
conv2_num_maps = 48
conv3_num_maps = 64
conv4_num_maps = 96
conv5_num_maps = 128
conv6_num_maps = 192
conv7_num_maps = 256
conv8_num_maps = 256
conv9_num_maps = 256
conv10_num_maps = 256
conv11_num_maps = 256
conv12_num_maps = 256
conv13_num_maps = 256
last_conv_flat_dim = 256*4
fc1_dim = 256
fc2_dim = 256
fc3_dim = 256
dropout_conv = True
activation = 'celu' # options: sigmoid, relu, l_relu, softplus, elu, celu, selu, tanh
alpha = 1/16


class Net(nn.Module):
    
    def __init__(self, num_classes):
        super(Net, self).__init__()
       
        self.conv1 = nn.Conv2d(input_channels_num, conv1_num_maps, kernel_size=(7,5), stride=(2)) # original kernel_size=(7,5)
        self.conv1_drop = nn.Dropout2d(0.05)
        self.conv1_bn = nn.BatchNorm2d(conv1_num_maps)
        
        self.conv2 = nn.Conv2d(conv1_num_maps, conv2_num_maps, kernel_size=(5,3), stride=(2))
        self.conv2_drop = nn.Dropout2d(0.1)
        self.conv2_bn = nn.BatchNorm2d(conv2_num_maps)
        
        self.conv3 = nn.Conv2d(conv2_num_maps, conv3_num_maps, kernel_size=(5,3), stride=(1))
        self.conv3_drop = nn.Dropout2d(0.15)
        self.conv3_bn = nn.BatchNorm2d(conv3_num_maps)
        
        self.conv4 = nn.Conv2d(conv3_num_maps, conv4_num_maps, kernel_size=(5,3), stride=(1))
        self.conv4_drop = nn.Dropout2d(0.2)
        self.conv4_bn = nn.BatchNorm2d(conv4_num_maps)
        
        self.conv5 = nn.Conv2d(conv4_num_maps, conv5_num_maps, kernel_size=(5,3), stride=(1))
        self.conv5_drop = nn.Dropout2d(0.25)
        self.conv5_bn = nn.BatchNorm2d(conv5_num_maps)
        
        self.conv6 = nn.Conv2d(conv5_num_maps, conv6_num_maps, kernel_size=(3,3), stride=(1))
        self.conv6_drop = nn.Dropout2d(0.3)
        self.conv6_bn = nn.BatchNorm2d(conv6_num_maps)
        
        self.conv7 = nn.Conv2d(conv6_num_maps, conv7_num_maps, kernel_size=(3,3), stride=(1))
        self.conv7_drop = nn.Dropout2d(0.35)
        self.conv7_bn = nn.BatchNorm2d(conv7_num_maps)
        
        self.conv8 = nn.Conv2d(conv7_num_maps, conv8_num_maps, kernel_size=(3,3), stride=(1))
        self.conv8_drop = nn.Dropout2d(0.4)
        self.conv8_bn = nn.BatchNorm2d(conv8_num_maps)
        
        self.conv9 = nn.Conv2d(conv8_num_maps, conv9_num_maps, kernel_size=(3,3), stride=(1))
        self.conv9_drop = nn.Dropout2d(0.45)
        self.conv9_bn = nn.BatchNorm2d(conv9_num_maps)  
        
        self.conv10 = nn.Conv2d(conv9_num_maps, conv10_num_maps, kernel_size=(3,3), stride=(1))
        self.conv10_drop = nn.Dropout2d(0.5)
        self.conv10_bn = nn.BatchNorm2d(conv10_num_maps)  
        
        self.conv11 = nn.Conv2d(conv10_num_maps, conv11_num_maps, kernel_size=(3,3), stride=(1))
        self.conv11_drop = nn.Dropout2d(0.5)
        self.conv11_bn = nn.BatchNorm2d(conv11_num_maps)
         
        self.conv12 = nn.Conv2d(conv11_num_maps, conv12_num_maps, kernel_size=(3,3), stride=(1))
        self.conv12_drop = nn.Dropout2d(0.5)
        self.conv12_bn = nn.BatchNorm2d(conv12_num_maps)
#         
        self.conv13 = nn.Conv2d(conv12_num_maps, conv13_num_maps, kernel_size=(2,2), stride=(1))
        self.conv13_drop = nn.Dropout2d(0.5)
        self.conv13_bn = nn.BatchNorm2d(conv13_num_maps)
        
        self.fc1 = nn.Linear(last_conv_flat_dim, fc1_dim)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc1_bn = nn.BatchNorm1d(fc1_dim)
        
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc2_drop = nn.Dropout(0.5)
        self.fc2_bn = nn.BatchNorm1d(fc2_dim)
        
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        self.fc3_drop = nn.Dropout(0.5)
        self.fc3_bn = nn.BatchNorm1d(fc3_dim)
        
        self.fc4 = nn.Linear(fc3_dim, num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv1_drop(x)
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv2_drop(x)
        
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv3_drop(x)
        
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv4_drop(x)
        
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv5_drop(x)
        
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv6_drop(x)
        
        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv7_drop(x)
        
        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv8_drop(x)
        
        x = self.conv9(x)
        x = self.conv9_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv9_drop(x)
        
        x = self.conv10(x)
        x = self.conv10_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv10_drop(x)
        
        x = self.conv11(x)
        x = self.conv11_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv11_drop(x)
        
        x = self.conv12(x)
        x = self.conv12_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv12_drop(x)
        
        x = self.conv13(x)
        x = self.conv13_bn(x)
        x_last_conv_flat = x.view(-1, last_conv_flat_dim) # Store flattened conv_5 layer
        x = ut.get_activated(x, activation, alph=alpha)
        if dropout_conv: x = self.conv13_drop(x)
        
        x = x.view(-1, last_conv_flat_dim)
        
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        x = self.fc1_drop(x)
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = ut.get_activated(x, activation, alph=alpha)
        x = self.fc2_drop(x)
        
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x_fc = x # Store fully connected layer
        x = ut.get_activated(x, activation, alph=alpha)
        x = self.fc3_drop(x)
        
        x = self.fc4(x)
        x_last_fc = x
        x = F.log_softmax(x, dim=1)
        
        return x, x_fc, x_last_conv_flat, x_last_fc
    
    
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

    if v.AUDIO_DATA_AUGMENTATION:
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomApply((transforms.RandomResizedCrop((input_height,input_width),scale=(0.9, 1.0),ratio=(1.0, 1.0)),), p=1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
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
            
    
    
    