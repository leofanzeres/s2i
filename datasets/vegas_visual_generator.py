import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torch
import datetime
import struct
import values as v


class VEGAS_VISUAL_GENERATOR(data.Dataset):
    """ VEGAS - Visually Engaged and Grounded AudioSet (AudioSet subset 10 sounds)
        http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/visual2sound.html

        Each record of this dataset consists of:
          self.label_length bits: string with class id + video (or spectrogram) id + frame (or spectrogram window) id
          self.input_length * 4 bits (float values) representing the net_audio embedding
          self.target_height * self.target_width * 3 (channels) bits representing the correspondent color image
          Size of each record in bits: self.label_length + self.input_length * 4 + self.target_height * self.target_width * 3
          
    Args:
        root (string): Root directory of dataset, i.e. where batches are stored.
        train (bool, optional): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    """

    train_data_size = 48945 # Entire train dataset: 48945
    test_data_size = 6825 # Entire val dataset: 5575 / Entire test dataset: 6825

    if v.AUDIO_EMBEDDING_DIMENSION < 2048:
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
                      'data_batch_066', 'data_batch_067', 'data_batch_068', 'data_batch_069', 'data_batch_070']

#         test_list = ['val_batch_001', 'val_batch_002', 'val_batch_003', 'val_batch_004', 'val_batch_005',
#                      'val_batch_006', 'val_batch_007', 'val_batch_008']

        test_list = ['test_batch_001', 'test_batch_002', 'test_batch_003', 'test_batch_004', 'test_batch_005',
                     'test_batch_006', 'test_batch_007', 'test_batch_008', 'test_batch_009', 'test_batch_010']
    else:
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
                      'data_batch_086', 'data_batch_087', 'data_batch_088', 'data_batch_089', 'data_batch_090',
                      'data_batch_091', 'data_batch_092', 'data_batch_093', 'data_batch_094', 'data_batch_095',
                      'data_batch_096', 'data_batch_097', 'data_batch_098']

#         test_list = ['val_batch_001', 'val_batch_002', 'val_batch_003', 'val_batch_004', 'val_batch_005',
#                      'val_batch_006', 'val_batch_007', 'val_batch_008', 'val_batch_009', 'val_batch_010',
#                      'val_batch_011', 'val_batch_012']

        test_list = ['test_batch_001', 'test_batch_002', 'test_batch_003', 'test_batch_004', 'test_batch_005',
                     'test_batch_006', 'test_batch_007', 'test_batch_008', 'test_batch_009', 'test_batch_010',
                     'test_batch_011', 'test_batch_012', 'test_batch_013', 'test_batch_014']

    def __init__(self, root, label_length, input_length, target_height, target_width, target_channels_num, train=True, transform=None, target_transform=None):

        self.root = root
        self.label_length = label_length
        self.input_length = input_length
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
                f += ".bin"
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                size = os.path.getsize(file)
                c = 0
                while c < (size/(self.label_length + self.input_length * 4 + self.target_flat_dimension)):
                    label = fo.read(self.label_length)

                    self.train_labels.append(label.decode('utf-8'))
                    c2 = 0
                    while c2 < self.input_length:
                        self.train_data.append(struct.unpack('f', fo.read(4)))
                        c2 += 1
                    c2 = 0
                    while c2 < self.target_flat_dimension:
                        b = fo.read(1)
                        self.train_targets.append(int.from_bytes(b, byteorder='little'))
                        c2 += 1
                    c += 1

                fo.close()

            print(str(datetime.datetime.now()) + ' ==> Finished loading train data')

            self.train_data = np.array(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data_size, self.input_length))
            self.train_targets = np.array(self.train_targets)
            self.train_targets = self.train_targets.reshape((self.train_data_size, self.target_height, self.target_width, self.target_channels_num))

            print(str(datetime.datetime.now()) + ' ==> Finished reshaping train data')

        else:
            self.test_data = []
            self.test_targets = []
            self.test_labels = []
            for f in self.test_list:
                f += ".bin"
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                size = os.path.getsize(file)
                c = 0
                while c < (size/(self.label_length + self.input_length * 4 + self.target_flat_dimension)):
                    label = fo.read(self.label_length)
                    self.test_labels.append(label.decode('utf-8'))
                    c2 = 0
                    while c2 < self.input_length:
                        self.test_data.append(struct.unpack('f', fo.read(4)))
                        c2 += 1
                    c2 = 0
                    while c2 < self.target_flat_dimension:
                        b = fo.read(1)
                        self.test_targets.append(int.from_bytes(b, byteorder='little'))
                        c2 += 1
                    c += 1

                fo.close()
            print(str(datetime.datetime.now()) + ' ==> Finished loading test data')
            self.test_data = np.array(self.test_data)
            self.test_data = self.test_data.reshape((self.test_data_size, self.input_length))
            self.test_targets = np.array(self.test_targets)
            self.test_targets = self.test_targets.reshape((self.test_data_size, self.target_height, self.target_width, self.target_channels_num))
            print(str(datetime.datetime.now()) + ' ==> Finished reshaping test data')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            audio_embeddings, target_images, labels = self.train_data[index], self.train_targets[index], self.train_labels[index]
        else:
            audio_embeddings, target_images, labels = self.test_data[index], self.test_targets[index], self.test_labels[index]

        audio_embeddings = torch.from_numpy(audio_embeddings).float()  # @UndefinedVariable

        if self.target_channels_num == 1:
            target_images = Image.fromarray(target_images.astype('uint8'), mode='L')
        else:
            target_images = Image.fromarray(target_images.astype('uint8'))

        if self.transform is not None:
            audio_embeddings = self.transform(audio_embeddings)

        if self.target_transform is not None:
            target_images = self.target_transform(target_images)

        return audio_embeddings, target_images, labels

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
