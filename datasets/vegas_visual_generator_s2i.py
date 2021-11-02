'''
Created on Jul 24, 2019

@author: leonardo
'''


import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torch
#from matplotlib import pyplot as plt
#import sys
import datetime
import struct
from matplotlib import pyplot as plt
import values as v

# if sys.version_info[0] == 2:
#     import cPickle as pickle
# else:
#     import pickle
# import pandas as pd


class VEGAS_VISUAL_GENERATOR_S2I(data.Dataset):
    """ VEGAS - Visually Engaged and Grounded AudioSet (AudioSet subset 10 sounds)
        http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/visual2sound.html

        Each record of this dataset consists of:
          self.label_length bits: string with class id + video (or spectrogram) id + frame (or spectrogram window) id
          self.input_height * self.input_width bits representing the correspondent spectrogram
          self.target_height * self.target_width * 3 (channels) bits representing the correspondent color image
          Size of each record in bits: self.label_length + self.input_height * self.input_width + self.target_height * self.target_width * 3

    Args:
        root (string): Root directory of dataset, i.e. where batches are stored.
        train (bool, optional): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    """

    train_data_size = 48945 # Entire dataset: 48945
    test_data_size = 6825 # Entire val dataset: 5575 / Entire test dataset: 6825

    train_list = ['data_batch_001', 'data_batch_002', 'data_batch_003', 'data_batch_004', 'data_batch_005',
                  'data_batch_006', 'data_batch_007', 'data_batch_008', 'data_batch_009', 'data_batch_010',
                  'data_batch_011', 'data_batch_012', 'data_batch_013', 'data_batch_014', 'data_batch_015',
                  'data_batch_016', 'data_batch_017', 'data_batch_018', 'data_batch_019', 'data_batch_020',
                  'data_batch_021', 'data_batch_022', 'data_batch_023', 'data_batch_024', 'data_batch_025',
                  'data_batch_026', 'data_batch_027', 'data_batch_028', 'data_batch_029', 'data_batch_030',
                  'data_batch_031', 'data_batch_032', 'data_batch_033', 'data_batch_034', 'data_batch_035',
                  'data_batch_036', 'data_batch_037', 'data_batch_038', 'data_batch_039', 'data_batch_040',
                  'data_batch_041', 'data_batch_042', 'data_batch_043', 'data_batch_044', 'data_batch_045',
                  'data_batch_046', 'data_batch_047', 'data_batch_048', 'data_batch_049', 'data_batch_050',
                  'data_batch_051', 'data_batch_052', 'data_batch_053', 'data_batch_054', 'data_batch_055',
                  'data_batch_056', 'data_batch_057', 'data_batch_058', 'data_batch_059', 'data_batch_060',
                  'data_batch_061', 'data_batch_062', 'data_batch_063', 'data_batch_064', 'data_batch_065',
                  'data_batch_066', 'data_batch_067', 'data_batch_068', 'data_batch_069', 'data_batch_070',
                  'data_batch_071', 'data_batch_072', 'data_batch_073', 'data_batch_074', 'data_batch_075',
                  'data_batch_076', 'data_batch_077', 'data_batch_078', 'data_batch_079', 'data_batch_080',
                  'data_batch_081', 'data_batch_082', 'data_batch_083', 'data_batch_084', 'data_batch_085',
                  'data_batch_086']


    # test_list = ['val_batch_001', 'val_batch_002', 'val_batch_003', 'val_batch_004', 'val_batch_005',
    #              'val_batch_006', 'val_batch_007', 'val_batch_008', 'val_batch_009', 'val_batch_010']
    test_list = ['test_batch_001', 'test_batch_002', 'test_batch_003', 'test_batch_004', 'test_batch_005',
                 'test_batch_006', 'test_batch_007', 'test_batch_008', 'test_batch_009', 'test_batch_010'
                 'test_batch_011', 'test_batch_012']

    def __init__(self, root, label_length, input_height, input_width, target_height, target_width, target_channels_num, train=True, transform=None, target_transform=None):

        #self.root = os.path.expanduser(root)
        self.root = root
        self.label_length = label_length
        self.input_height = input_height
        self.input_width = input_width
        self.input_flat_dimension = input_height * input_width
        self.target_height = target_height
        self.target_width = target_width
        self.target_channels_num= target_channels_num
        self.target_flat_dimension = target_height * target_width * target_channels_num
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set



        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_targets = []
            self.train_labels = []
            for f in self.train_list:
                #f = fentry[0]
                # file = os.path.join(self.root, self.base_folder, f)
                f += ".bin"
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                size = os.path.getsize(file)
                c = 0
                while c < (size/(self.label_length + self.input_flat_dimension + self.target_flat_dimension)):
                    #fo.seak(1,1)
                    label = fo.read(self.label_length) # class_id + video_id + window_id (without letters)
                    #print('Loading: ' + str(int.from_bytes(label, byteorder='little')))
                    #self.train_labels = int.from_bytes(label, byteorder='little')
                    #self.train_labels.append(int.from_bytes(label, byteorder='little'))
                    self.train_labels.append(label.decode('utf-8'))
                    c2 = 0
                    while c2 < self.input_flat_dimension:
                        b = fo.read(1)
                        #b = fo.read(4)
                        #self.train_data.append(struct.unpack('f', b))
                        self.train_data.append(int.from_bytes(b, byteorder='little'))
                        c2 += 1
                    c2 = 0
                    while c2 < self.target_flat_dimension:
                        b = fo.read(1)
                        #self.train_targets.append(struct.unpack('f', b))
                        self.train_targets.append(int.from_bytes(b, byteorder='little'))
                        c2 += 1
                    c += 1
#                     img = np.array(self.train_data)
#                     img = img.reshape((200,200))
#                     plt.imshow(img)
#                     plt.show()
                # Using Pickle
                # entry = pickle.load(fo, encoding='latin1') # Only for Python 3
                # self.train_data.append(entry['data'])
                # self.train_labels += entry['labels']

                fo.close()

                # Plot all vectors of the batch as a single image
#                 train_data_visual = np.array(self.train_data)
#                 train_data_visual = train_data_visual.reshape((1200, self.input_length))
#                 train_data_visual = (train_data_visual - train_data_visual.min()) / (train_data_visual.max() - train_data_visual.min())
#                 train_data_visual = Image.fromarray((train_data_visual * 255).astype(np.uint8))
#                 train_targets_visual = np.array(self.train_targets)
#                 train_targets_visual = train_targets_visual.reshape((1200, self.input_length))
#                 train_targets_visual = (train_targets_visual - train_targets_visual.min()) / (train_targets_visual.max() - train_targets_visual.min())
#                 train_targets_visual = Image.fromarray((train_targets_visual * 255).astype(np.uint8))
#                 train_data_visual.save('train_data_visual.jpg')
#                 plt.imshow(train_data_visual)
#                 plt.show()
#                 train_targets_visual.save('train_targets_visual.jpg')
#                 plt.imshow(train_targets_visual)
#                 plt.show()

            print(str(datetime.datetime.now()) + ' ==> Finished loading train data')
            #self.train_data = np.concatenate(self.train_data)
            #self.train_labels = np.array(self.train_labels)
            #self.train_labels = self.train_labels.reshape((1200, 11)) # 55081
            self.train_data = np.array(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data_size, self.input_height, self.input_width)) # 48945
            self.train_targets = np.array(self.train_targets)
            self.train_targets = self.train_targets.reshape((self.train_data_size, self.target_height, self.target_width, self.target_channels_num)) # 48945
            # Plot image
            # https://stackoverflow.com/questions/3823752/display-image-as-grayscale-using-matplotlib?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
#             for i in self.train_data:
#                 img = np.array(i)
#                 plt.imshow(img, cmap='gray', vmin = 0, vmax = 255) # without pixel intensity adjustment
#                 # plt.imshow(img, cmap='gray') # with pixel intensity adjustment
#                 # plt.imshow(img, cmap='gray_r', vmin = 0, vmax = 255) # Inverse
#                 plt.show()

            #self.train_data = self.train_data.transpose((0, 2, 1))  # convert
            # Using color images
            #self.train_data = self.train_data.reshape((600, 3, 200, 200))
            #self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            print(str(datetime.datetime.now()) + ' ==> Finished reshaping train data')


        else:
            self.test_data = []
            self.test_targets = []
            self.test_labels = []
            for f in self.test_list:
                #f = self.test_list[0][0]
                #le = os.path.join(self.root, self.base_folder, f)
                f += ".bin"
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                size = os.path.getsize(file)
                c = 0
                while c < (size/(self.label_length + self.input_flat_dimension + self.target_flat_dimension)):
                    #fo.seak(1,1)
                    label = fo.read(self.label_length) # class_id + video_id + window_id (without letters)
                    #print('Loading: ' + str(int.from_bytes(label, byteorder='little')))
                    #self.test_labels = int.from_bytes(label, byteorder='little')
                    #self.test_labels.append(int.from_bytes(label, byteorder='little'))
                    self.test_labels.append(label.decode('utf-8'))
                    c2 = 0
                    while c2 < self.input_flat_dimension:
                        b = fo.read(1)
                        self.test_data.append(int.from_bytes(b, byteorder='little'))
                        c2 += 1
                    c2 = 0
                    while c2 < self.target_flat_dimension:
                        b = fo.read(1)
                        self.test_targets.append(int.from_bytes(b, byteorder='little'))
                        c2 += 1
                    c += 1
                # Using Pickle
                # entry = pickle.load(fo, encoding='latin1') # Only for Python 3
                # self.test_data = entry['data']
                # self.test_labels = entry['labels']

                fo.close()
            print(str(datetime.datetime.now()) + ' ==> Finished loading test data')
            self.test_data = np.array(self.test_data)
            self.test_data = self.test_data.reshape((self.test_data_size, self.input_height, self.input_width))
            self.test_targets = np.array(self.test_targets)
            self.test_targets = self.test_targets.reshape((self.test_data_size, self.target_height, self.target_width, self.target_channels_num)) #  # Val: 5575 / Test: 6825
            print(str(datetime.datetime.now()) + ' ==> Finished reshaping test data')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            input_images, target_images, labels = self.train_data[index], self.train_targets[index], self.train_labels[index]
        else:
            input_images, target_images, labels = self.test_data[index], self.test_targets[index], self.test_labels[index]

        input_images = Image.fromarray(input_images.astype('uint8'), mode='L') # when loading images from an array

        if self.target_channels_num == 1:
            target_images = Image.fromarray(target_images.astype('uint8'), mode='L') # when loading images from an array
        else:
            target_images = Image.fromarray(target_images.astype('uint8')) # when loading images from an array

        if self.transform is not None:
            input_images = self.transform(input_images)

        if self.target_transform is not None:
            target_images = self.target_transform(target_images)

        return input_images, target_images, labels

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
