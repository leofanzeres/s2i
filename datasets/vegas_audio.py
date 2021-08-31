import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import datetime


class VEGAS_AUDIO(data.Dataset):
    """ VEGAS - Visually Engaged and Grounded AudioSet (AudioSet subset 10 sounds)
        http://bvision11.cs.unc.edu/bigpen/yipin/visual2sound_webpage/visual2sound.html
    """
    train_data_size = 49435 # Entire train dataset: 49435
    test_data_size = 6920 # Entire val dataset: 5635 / Entire test dataset: 6920
    
    train_list = ['data_batch_001', 'data_batch_002', 'data_batch_003', 'data_batch_004', 'data_batch_005',
                  'data_batch_006', 'data_batch_007', 'data_batch_008', 'data_batch_009', 'data_batch_010',
                  'data_batch_011', 'data_batch_012', 'data_batch_013', 'data_batch_014', 'data_batch_015',
                  'data_batch_016', 'data_batch_017', 'data_batch_018', 'data_batch_019', 'data_batch_020', 
                  'data_batch_021', 'data_batch_022', 'data_batch_023', 'data_batch_024', 'data_batch_025',
                  'data_batch_026', 'data_batch_027', 'data_batch_028']
    
    #test_list = ['val_batch_001', 'val_batch_002', 'val_batch_003', 'val_batch_004']
    test_list = ['test_batch_001', 'test_batch_002', 'test_batch_003', 'test_batch_004']
    

    def __init__(self, root, input_height, input_width, input_channels_num, train=True, transform=None, target_transform=None):
        
        self.root = root
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels_num = input_channels_num
        self.input_flat_dimension = input_height * input_width * input_channels_num
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for f in self.train_list:
                f += ".bin"
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                size = os.path.getsize(file)
                c = 0
                while c < (size/(self.input_flat_dimension + 1)):
                    label = fo.read(1)
                    self.train_labels.append(int.from_bytes(label, byteorder='little'))
                    c2 = 0
                    while c2 < self.input_flat_dimension:
                        b = fo.read(1)
                        self.train_data.append(int.from_bytes(b, byteorder='little'))
                        c2 += 1
                    c += 1
                fo.close()
            print(str(datetime.datetime.now()) + ' ==> Finished loading train data')
            self.train_data = np.array(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data_size, self.input_height, self.input_width))
            
            print(str(datetime.datetime.now()) + ' ==> Finished reshaping train data')
            
        else:
            self.test_data = []
            self.test_labels = []
            for f in self.test_list:
                f += ".bin"
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                size = os.path.getsize(file)
                c = 0
                while c < (size/(self.input_flat_dimension + 1)):
                    label = fo.read(1)
                    self.test_labels.append(int.from_bytes(label, byteorder='little'))
                    c2 = 0
                    while c2 < self.input_flat_dimension:
                        b = fo.read(1)
                        self.test_data.append(int.from_bytes(b, byteorder='little'))
                        c2 += 1
                    c += 1
                fo.close()
            print(str(datetime.datetime.now()) + ' ==> Finished loading test data')
            self.test_data = np.array(self.test_data)
            self.test_data = self.test_data.reshape((self.test_data_size, self.input_height, self.input_width))

            print(str(datetime.datetime.now()) + ' ==> Finished reshaping test data')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.input_channels_num == 1:
            img = Image.fromarray(img.astype('uint8'), mode='L')
        else:
            img = Image.fromarray(img.astype('uint8'))
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


