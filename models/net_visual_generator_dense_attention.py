'''
Created on Feb 1, 2019

@author: leonardo

'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.modules as nnmodules
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.module import _addindent
import torchvision.transforms as transforms
import math
import numpy as np
import sys
import argparse
import utils as ut  # @UnresolvedImport
import values as v
if v.AUDIO_DATA_AUGMENTATION | v.TRAIN_AUDIO_ENCODER | (v.AUDIO_ENCODER_LAYER < 13):
    import datasets.vegas_visual_generator_s2i as vegas_visual_generator_s2i
else:
    import datasets.vegas_visual_generator as vegas_visual_generator


parser = argparse.ArgumentParser(description='PyTorch VEGAS_AUDIO_10 Training')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # @UndefinedVariable
audio_embedding_dim = v.AUDIO_EMBEDDING_DIMENSION
input_height = 128
input_width = 100
target_height = 96
target_width = 96
target_channels_num = 3
label_length = 11

if (v.USE_BRIDGE | v.MAKE_PRIME | v.DEBLUR | v.USE_TRANSCODER):
    input_embedding_dim = v.VISUAL_EMBEDDING_DIMENSION
else:
    input_embedding_dim = v.AUDIO_EMBEDDING_DIMENSION
noise_vector_dim = v.NOISE_VECTOR_DIM

batchnorm = True
dropout = (1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0)


class Net(nn.Module):
    
    def __init__(self, num_classes, embedding_dimension, activation = 'relu', activation_alpha=1.0):
        super(Net, self).__init__()
        # Get dimensions
        (input_feature_map_shape,input_channels_num,convT1_num_maps,convT2_num_maps,convT3_num_maps,convT4_num_maps,convT5_num_maps,convT6_num_maps,
            convT7_num_maps,convT8_num_maps,convT9_num_maps,convT10_num_maps,convT11_num_maps,convT12_num_maps,convT13_num_maps,convT14_num_maps,
            convT15_num_maps,convT16_num_maps,convT17_num_maps,convT18_num_maps,convT1_to_17_shape,transition_shape,encoder_maps_shape) = get_dimensions(embedding_dimension)
            
        self.input_feature_map_shape = input_feature_map_shape
        self.input_channels_num = input_channels_num
        if activation == 'custom':
            self.activation = ['celu']*24
            self.activation_alpha = [0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,4.0,4.0,4.0,4.0]
        else:
            self.activation = [activation]*24 # activation options: sigmoid, relu, l_relu, elu, celu, selu, tanh
            self.activation_alpha = [activation_alpha]*24        
                 
        # Transition 0
        if input_feature_map_shape[0] == 1:
            self.transition0 = nn.ConvTranspose2d(in_channels = input_channels_num, out_channels = convT1_num_maps, kernel_size = (2,2), stride = 1, padding=0)
        elif input_feature_map_shape[0] == 2:
            self.transition0 = nn.ConvTranspose2d(in_channels = input_channels_num, out_channels = convT1_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.transition0_bn = nn.BatchNorm2d(convT1_num_maps)
        if dropout[0]>0: self.transition0_drop = nn.Dropout2d(0.15)

        # Attention Block 0
        if (v.NUM_ATTENTION_BLOCKS[0] > 0) or (v.SHARE_AUDIO_EMB and v.ATTENTIONAL_NET):
            if v.TRANSPOSE_EMB == (0,0): # Channel Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_seq_len = transition_shape[0][0] * v.PATCHSIZE[0] * v.PATCHSIZE[0]
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_emb_dimension = (transition_shape[0][1] * transition_shape[0][2]) // (v.PATCHSIZE[0] * v.PATCHSIZE[0])
            else: # Spatial Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_seq_len = (transition_shape[0][1] * transition_shape[0][2]) // (v.PATCHSIZE[0] * v.PATCHSIZE[0])
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_emb_dimension = transition_shape[0][0] * v.PATCHSIZE[0] * v.PATCHSIZE[0]

            d_model = v.D_MODEL[0]
            self.src_embed_proj_0 = nn.Linear(src_emb_dimension, d_model)
            self.tgt_embed_proj_0 = nn.Linear(tgt_emb_dimension, d_model)
            self.out_embed_proj_0 = nn.Linear(d_model, tgt_emb_dimension)

            # Create the decoder blocks
            decoder_blocks_0 = []
            if v.SHARE_ATTENTION_BLOCKS:
                decoder_self_attention_block_0 = MultiHeadAttentionBlock(d_model, v.HEADS[0], v.DROPOUT_ATT[0])
                decoder_cross_attention_block_0 = MultiHeadAttentionBlock(d_model, v.HEADS[0], v.DROPOUT_ATT[0])
            for _ in range(v.NUM_ATTENTION_BLOCKS[0]):
                if not v.SHARE_ATTENTION_BLOCKS:
                    decoder_self_attention_block_0 = MultiHeadAttentionBlock(d_model, v.HEADS[0], v.DROPOUT_ATT[0])
                    decoder_cross_attention_block_0 = MultiHeadAttentionBlock(d_model, v.HEADS[0], v.DROPOUT_ATT[0])
                feed_forward_block_0 = FeedForwardBlock(d_model, d_model*4, v.DROPOUT_ATT[0])
                decoder_block_0 = DecoderBlock(d_model, decoder_self_attention_block_0, decoder_cross_attention_block_0, feed_forward_block_0, v.DROPOUT_ATT[0])
                decoder_blocks_0.append(decoder_block_0)
            if (v.NUM_ATTENTION_BLOCKS[0] == 0) and (v.NUM_ATTENTION_BLOCKS[1] > 0): # TODO Need to fix this, because it doesn't work when v.NUM_ATTENTION_BLOCKS[0] and v.NUM_ATTENTION_BLOCKS[1] are both greater than zero but different
                for _ in range(v.NUM_ATTENTION_BLOCKS[1]):
                    if not v.SHARE_ATTENTION_BLOCKS:
                        decoder_self_attention_block_0 = MultiHeadAttentionBlock(d_model, v.HEADS[0], v.DROPOUT_ATT[0])
                        decoder_cross_attention_block_0 = MultiHeadAttentionBlock(d_model, v.HEADS[0], v.DROPOUT_ATT[0])
                    feed_forward_block_0 = FeedForwardBlock(d_model, d_model*4, v.DROPOUT_ATT[0])
                    decoder_block_0 = DecoderBlock(d_model, decoder_self_attention_block_0, decoder_cross_attention_block_0, feed_forward_block_0, v.DROPOUT_ATT[0])
                    decoder_blocks_0.append(decoder_block_0)
            
            # Create the decoder
            self.decoder_0 = Decoder(d_model, nn.ModuleList(decoder_blocks_0))

            if (v.NUM_ATTENTION_BLOCKS[0] > 0):
                if v.ADD_POS_ENCOD:
                    self.src_pos_0 = PositionalEnconding(d_model, src_seq_len, 0.0)
                    self.tgt_pos_0 = PositionalEnconding(d_model, tgt_seq_len, 0.0)
                else:
                    self.src_pos_0 = None
                    self.tgt_pos_0 = None
            else: # To be used in Attention Block 1
                if v.ADD_POS_ENCOD:
                    self.src_pos_0 = PositionalEnconding(d_model, src_seq_len, 0.0)
                else:
                    self.src_pos_0 = None

        
        # Dense Block 1
        
        self.convT1 = nn.ConvTranspose2d(in_channels = convT1_num_maps, out_channels = convT1_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT1_bn = nn.BatchNorm2d(convT1_num_maps)
        if dropout[1]>0: self.convT1_drop = nn.Dropout2d(0.0)
        
        self.convT2 = nn.ConvTranspose2d(in_channels = convT1_num_maps, out_channels = convT2_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT2_bn = nn.BatchNorm2d(convT2_num_maps)
        if dropout[2]>0: self.convT2_drop = nn.Dropout2d(0.0)
        
        self.convT3 = nn.ConvTranspose2d(in_channels = convT1_num_maps+convT2_num_maps, out_channels = convT3_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT3_bn = nn.BatchNorm2d(convT3_num_maps)
        if dropout[3]>0: self.convT3_drop = nn.Dropout2d(0.0)
        
        # Transition 1
        self.transition1 = nn.ConvTranspose2d(in_channels = convT1_num_maps+convT2_num_maps+convT3_num_maps, out_channels = convT3_num_maps, kernel_size = (3,3), stride = 1, padding=0)
        if batchnorm: self.transition1_bn = nn.BatchNorm2d(convT3_num_maps)
        if dropout[4]>0: self.transition1_drop = nn.Dropout2d(0.13)
        
        # Attention Block 1
        if v.NUM_ATTENTION_BLOCKS[1] > 0:
            if v.TRANSPOSE_EMB == (0,0): # Channel Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_seq_len = transition_shape[1][0] * v.PATCHSIZE[1] * v.PATCHSIZE[1]
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_emb_dimension = (transition_shape[1][1] * transition_shape[1][2]) // (v.PATCHSIZE[1] * v.PATCHSIZE[1])
            else: # Spatial Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_seq_len = (transition_shape[1][1] * transition_shape[1][2]) // (v.PATCHSIZE[1] * v.PATCHSIZE[1])
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_emb_dimension = transition_shape[1][0] * v.PATCHSIZE[1] * v.PATCHSIZE[1]
            
            d_model = v.D_MODEL[1]
            if not v.SHARE_AUDIO_EMB: self.src_embed_proj_1 = nn.Linear(src_emb_dimension, d_model) # concatenated source embedding: nn.Linear(encoder_maps_shape[10][1] * encoder_maps_shape[10][2] + encoder_maps_shape[11][1] * encoder_maps_shape[11][2], d_model)
            # self.tgt_embed_proj_1 = nn.Linear(tgt_emb_dimension, d_model)
            # self.out_embed_proj_1 = nn.Linear(d_model, tgt_emb_dimension)

            # # Create the decoder blocks
            # decoder_blocks_1 = []
            # if v.SHARE_ATTENTION_BLOCKS:
            #     decoder_self_attention_block_1 = MultiHeadAttentionBlock(d_model, v.HEADS[1], v.DROPOUT_ATT[4])
            #     decoder_cross_attention_block_1 = MultiHeadAttentionBlock(d_model, v.HEADS[1], v.DROPOUT_ATT[4])
            # for _ in range(v.NUM_ATTENTION_BLOCKS[1]):
            #     if not v.SHARE_ATTENTION_BLOCKS:
            #         decoder_self_attention_block_1 = MultiHeadAttentionBlock(d_model, v.HEADS[1], v.DROPOUT_ATT[4])
            #         decoder_cross_attention_block_1 = MultiHeadAttentionBlock(d_model, v.HEADS[1], v.DROPOUT_ATT[4])
            #     feed_forward_block_1 = FeedForwardBlock(d_model, d_model*4, v.DROPOUT_ATT[4])
            #     decoder_block_1 = DecoderBlock(d_model, decoder_self_attention_block_1, decoder_cross_attention_block_1, feed_forward_block_1, v.DROPOUT_ATT[4])
            #     decoder_blocks_1.append(decoder_block_1)
            
            # # Create the decoder
            # self.decoder_1 = Decoder(d_model, nn.ModuleList(decoder_blocks_1))
                    
            if v.ADD_POS_ENCOD:
                if not v.SHARE_AUDIO_EMB: self.src_pos_1 = PositionalEnconding(d_model, src_seq_len, 0.0)
                self.tgt_pos_1 = PositionalEnconding(d_model, tgt_seq_len, 0.0)
            else:
                if not v.SHARE_AUDIO_EMB: self.src_pos_1 = None
                self.tgt_pos_1 = None

        
        # Dense Block 2
        self.convT4 = nn.ConvTranspose2d(in_channels = convT3_num_maps, out_channels = convT4_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT4_bn = nn.BatchNorm2d(convT4_num_maps)
        if dropout[5]>0: self.convT4_drop = nn.Dropout2d(0.0)
        
        self.convT5 = nn.ConvTranspose2d(in_channels = convT4_num_maps, out_channels = convT5_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT5_bn = nn.BatchNorm2d(convT5_num_maps)
        if dropout[6]>0: self.convT5_drop = nn.Dropout2d(0.0)
        
        self.convT6 = nn.ConvTranspose2d(in_channels = convT4_num_maps+convT5_num_maps, out_channels = convT6_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT6_bn = nn.BatchNorm2d(convT6_num_maps)
        if dropout[7]>0: self.convT6_drop = nn.Dropout2d(0.0)
        
        # Transition 2
        self.transition2 = nn.ConvTranspose2d(in_channels = convT4_num_maps+convT5_num_maps+convT6_num_maps, out_channels = convT6_num_maps, kernel_size = (3,3), stride = 1, padding=0)
        if batchnorm: self.transition2_bn = nn.BatchNorm2d(convT6_num_maps)
        if dropout[8]>0: self.transition2_drop = nn.Dropout2d(0.11)
        
        # Attention Block 2
        if v.NUM_ATTENTION_BLOCKS[2] > 0:
            if v.TRANSPOSE_EMB == (0,0): # Channel Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_seq_len = transition_shape[2][0] * v.PATCHSIZE[2] * v.PATCHSIZE[2]
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_emb_dimension = (transition_shape[2][1] * transition_shape[2][2]) // (v.PATCHSIZE[2] * v.PATCHSIZE[2])
            else: # Spatial Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_seq_len = (transition_shape[2][1] * transition_shape[2][2]) // (v.PATCHSIZE[2] * v.PATCHSIZE[2])
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_emb_dimension = transition_shape[2][0] * v.PATCHSIZE[2] * v.PATCHSIZE[2]

            d_model = v.D_MODEL[2]
            if not v.SHARE_AUDIO_EMB: self.src_embed_proj_2 = nn.Linear(src_emb_dimension, d_model) # concatenated source embedding: ... 
            self.tgt_embed_proj_2 = nn.Linear(tgt_emb_dimension, d_model)
            self.out_embed_proj_2 = nn.Linear(d_model, tgt_emb_dimension)

            # Create the decoder blocks
            decoder_blocks_2 = []
            if v.SHARE_ATTENTION_BLOCKS:
                decoder_self_attention_block_2 = MultiHeadAttentionBlock(d_model, v.HEADS[2], v.DROPOUT_ATT[8])
                decoder_cross_attention_block_2 = MultiHeadAttentionBlock(d_model, v.HEADS[2], v.DROPOUT_ATT[8])
            for _ in range(v.NUM_ATTENTION_BLOCKS[2]):
                if not v.SHARE_ATTENTION_BLOCKS:
                    decoder_self_attention_block_2 = MultiHeadAttentionBlock(d_model, v.HEADS[2], v.DROPOUT_ATT[8])
                    decoder_cross_attention_block_2 = MultiHeadAttentionBlock(d_model, v.HEADS[2], v.DROPOUT_ATT[8])
                feed_forward_block_2 = FeedForwardBlock(d_model, d_model*4, v.DROPOUT_ATT[8])
                decoder_block_2 = DecoderBlock(d_model, decoder_self_attention_block_2, decoder_cross_attention_block_2, feed_forward_block_2, v.DROPOUT_ATT[8])
                decoder_blocks_2.append(decoder_block_2)
            
            # Create the decoder
            self.decoder_2 = Decoder(d_model, nn.ModuleList(decoder_blocks_2))
                    
            if v.ADD_POS_ENCOD:
                if not v.SHARE_AUDIO_EMB: self.src_pos_2 = PositionalEnconding(d_model, src_seq_len, 0.0)
                self.tgt_pos_2 = PositionalEnconding(d_model, tgt_seq_len, 0.0)
            else:
                if not v.SHARE_AUDIO_EMB: self.src_pos_2 = None
                self.tgt_pos_2 = None

        
        # Dense Block 3
        
        self.convT7 = nn.ConvTranspose2d(in_channels = convT6_num_maps, out_channels = convT7_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT7_bn = nn.BatchNorm2d(convT7_num_maps)
        if dropout[9]>0: self.convT7_drop = nn.Dropout2d(0.0)
        
        self.convT8 = nn.ConvTranspose2d(in_channels = convT7_num_maps, out_channels = convT8_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT8_bn = nn.BatchNorm2d(convT8_num_maps)
        if dropout[10]>0: self.convT8_drop = nn.Dropout2d(0.0)
        
        self.convT9 = nn.ConvTranspose2d(in_channels = convT7_num_maps+convT8_num_maps, out_channels = convT9_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT9_bn = nn.BatchNorm2d(convT9_num_maps)
        if dropout[11]>0: self.convT9_drop = nn.Dropout2d(0.0)
        
        # Transition 3
        self.transition3 = nn.ConvTranspose2d(in_channels = convT7_num_maps+convT8_num_maps+convT9_num_maps, out_channels = convT9_num_maps, kernel_size = (4,4), stride = 2, padding=1)
        if batchnorm: self.transition3_bn = nn.BatchNorm2d(convT9_num_maps)
        if dropout[12]>0: self.transition3_drop = nn.Dropout2d(0.09)

        # Attention Block 3
        if v.NUM_ATTENTION_BLOCKS[3] > 0:
            if v.TRANSPOSE_EMB == (0,0): # Channel Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_seq_len = transition_shape[3][0] * v.PATCHSIZE[3] * v.PATCHSIZE[3]
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_emb_dimension = (transition_shape[3][1] * transition_shape[3][2]) // (v.PATCHSIZE[3] * v.PATCHSIZE[3])
            else: # Spatial Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_seq_len = (transition_shape[3][1] * transition_shape[3][2]) // (v.PATCHSIZE[3] * v.PATCHSIZE[3])
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_emb_dimension = transition_shape[3][0] * v.PATCHSIZE[3] * v.PATCHSIZE[3]

            d_model = v.D_MODEL[3]
            if not v.SHARE_AUDIO_EMB: self.src_embed_proj_3 = nn.Linear(src_emb_dimension, d_model) # concatenated source embedding: ... 
            self.tgt_embed_proj_3 = nn.Linear(tgt_emb_dimension, d_model)
            self.out_embed_proj_3 = nn.Linear(d_model, tgt_emb_dimension)

            # Create the decoder blocks
            decoder_blocks_3 = []
            if v.SHARE_ATTENTION_BLOCKS:
                decoder_self_attention_block_3 = MultiHeadAttentionBlock(d_model, v.HEADS[3], v.DROPOUT_ATT[12])
                decoder_cross_attention_block_3 = MultiHeadAttentionBlock(d_model, v.HEADS[3], v.DROPOUT_ATT[12])
            for _ in range(v.NUM_ATTENTION_BLOCKS[3]):
                if not v.SHARE_ATTENTION_BLOCKS:
                    decoder_self_attention_block_3 = MultiHeadAttentionBlock(d_model, v.HEADS[3], v.DROPOUT_ATT[12])
                    decoder_cross_attention_block_3 = MultiHeadAttentionBlock(d_model, v.HEADS[3], v.DROPOUT_ATT[12])
                feed_forward_block_3 = FeedForwardBlock(d_model, d_model*4, v.DROPOUT_ATT[12])
                decoder_block_3 = DecoderBlock(d_model, decoder_self_attention_block_3, decoder_cross_attention_block_3, feed_forward_block_3, v.DROPOUT_ATT[12])
                decoder_blocks_3.append(decoder_block_3)
            
            # Create the decoder
            self.decoder_3 = Decoder(d_model, nn.ModuleList(decoder_blocks_3))            
                    
            if v.ADD_POS_ENCOD:
                if not v.SHARE_AUDIO_EMB: self.src_pos_3 = PositionalEnconding(d_model, src_seq_len, 0.0)
                self.tgt_pos_3 = PositionalEnconding(d_model, tgt_seq_len, 0.0)
            else:
                if not v.SHARE_AUDIO_EMB: self.src_pos_3 = None
                self.tgt_pos_3 = None

        
        # Dense Block 4    
        
        self.convT10 = nn.ConvTranspose2d(in_channels = convT9_num_maps, out_channels = convT10_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT10_bn = nn.BatchNorm2d(convT10_num_maps)
        if dropout[13]>0: self.convT10_drop = nn.Dropout2d(0.0)
        
        self.convT11 = nn.ConvTranspose2d(in_channels = convT10_num_maps, out_channels = convT11_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT11_bn = nn.BatchNorm2d(convT11_num_maps)
        if dropout[14]>0: self.convT11_drop = nn.Dropout2d(0.0)
        
        self.convT12 = nn.ConvTranspose2d(in_channels = convT10_num_maps+convT11_num_maps, out_channels = convT12_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT12_bn = nn.BatchNorm2d(convT12_num_maps)
        if dropout[15]>0: self.convT12_drop = nn.Dropout2d(0.0)
        
        # Transition 4
        self.transition4 = nn.ConvTranspose2d(in_channels = convT10_num_maps+convT11_num_maps+convT12_num_maps, out_channels = convT12_num_maps, kernel_size = (4,4), stride = 2, padding=1)
        if batchnorm: self.transition4_bn = nn.BatchNorm2d(convT12_num_maps)
        if dropout[16]>0: self.transition4_drop = nn.Dropout2d(0.07)
        
        # Attention Block 4
        if v.NUM_ATTENTION_BLOCKS[4] > 0:
            if v.TRANSPOSE_EMB == (0,0): # Channel Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_seq_len = transition_shape[4][0] * v.PATCHSIZE[4] * v.PATCHSIZE[4]
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_emb_dimension = (transition_shape[4][1] * transition_shape[4][2]) // (v.PATCHSIZE[4] * v.PATCHSIZE[4])
            else: # Spatial Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_seq_len = (transition_shape[4][1] * transition_shape[4][2]) // (v.PATCHSIZE[4] * v.PATCHSIZE[4])
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_emb_dimension = transition_shape[4][0] * v.PATCHSIZE[4] * v.PATCHSIZE[4]

            d_model = v.D_MODEL[4]
            if not v.SHARE_AUDIO_EMB: self.src_embed_proj_4 = nn.Linear(src_emb_dimension, d_model) # concatenated source embedding: ... 
            self.tgt_embed_proj_4 = nn.Linear(tgt_emb_dimension, d_model)
            self.out_embed_proj_4 = nn.Linear(d_model, tgt_emb_dimension)

            # Create the decoder blocks
            decoder_blocks_4 = []
            if v.SHARE_ATTENTION_BLOCKS:
                decoder_self_attention_block_4 = MultiHeadAttentionBlock(d_model, v.HEADS[4], v.DROPOUT_ATT[16])
                decoder_cross_attention_block_4 = MultiHeadAttentionBlock(d_model, v.HEADS[4], v.DROPOUT_ATT[16])
            for _ in range(v.NUM_ATTENTION_BLOCKS[4]):
                if not v.SHARE_ATTENTION_BLOCKS:
                    decoder_self_attention_block_4 = MultiHeadAttentionBlock(d_model, v.HEADS[4], v.DROPOUT_ATT[16])
                    decoder_cross_attention_block_4 = MultiHeadAttentionBlock(d_model, v.HEADS[4], v.DROPOUT_ATT[16])
                feed_forward_block_4 = FeedForwardBlock(d_model, d_model*4, v.DROPOUT_ATT[16])
                decoder_block_4 = DecoderBlock(d_model, decoder_self_attention_block_4, decoder_cross_attention_block_4, feed_forward_block_4, v.DROPOUT_ATT[16])
                decoder_blocks_4.append(decoder_block_4)
            
            # Create the decoder
            self.decoder_4 = Decoder(d_model, nn.ModuleList(decoder_blocks_4))            
                 
            if v.ADD_POS_ENCOD:
                if not v.SHARE_AUDIO_EMB: self.src_pos_4 = PositionalEnconding(d_model, src_seq_len, 0.0)
                self.tgt_pos_4 = PositionalEnconding(d_model, tgt_seq_len, 0.0)
            else:
                if not v.SHARE_AUDIO_EMB: self.src_pos_4 = None
                self.tgt_pos_4 = None

        
        # Dense Block 5 
        
        self.convT13 = nn.ConvTranspose2d(in_channels = convT12_num_maps, out_channels = convT13_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT13_bn = nn.BatchNorm2d(convT13_num_maps)
        if dropout[17]>0: self.convT13_drop = nn.Dropout2d(0.0)
        
        self.convT14 = nn.ConvTranspose2d(in_channels = convT13_num_maps, out_channels = convT14_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT14_bn = nn.BatchNorm2d(convT14_num_maps)
        if dropout[18]>0: self.convT14_drop = nn.Dropout2d(0.0)
        
        self.convT15 = nn.ConvTranspose2d(in_channels = convT13_num_maps+convT14_num_maps, out_channels = convT15_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT15_bn = nn.BatchNorm2d(convT15_num_maps)
        if dropout[19]>0: self.convT15_drop = nn.Dropout2d(0.0)
        
        # Transition 5
        self.transition5 = nn.ConvTranspose2d(in_channels = convT13_num_maps+convT14_num_maps+convT15_num_maps, out_channels = convT15_num_maps, kernel_size = (4,4), stride = 2, padding=1)
        if batchnorm: self.transition5_bn = nn.BatchNorm2d(convT15_num_maps)
        if dropout[20]>0: self.transition5_drop = nn.Dropout2d(0.05)
        
        # Attention Block 5
        if v.NUM_ATTENTION_BLOCKS[5] > 0:
            if v.TRANSPOSE_EMB == (0,0): # Channel Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_seq_len = transition_shape[5][0] * v.PATCHSIZE[5] * v.PATCHSIZE[5]
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_emb_dimension = (transition_shape[5][1] * transition_shape[5][2]) // (v.PATCHSIZE[5] * v.PATCHSIZE[5])
            else: # Spatial Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_seq_len = (transition_shape[5][1] * transition_shape[5][2]) // (v.PATCHSIZE[5] * v.PATCHSIZE[5])
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_emb_dimension = transition_shape[5][0] * v.PATCHSIZE[5] * v.PATCHSIZE[5]

            d_model = v.D_MODEL[5]
            if not v.SHARE_AUDIO_EMB: self.src_embed_proj_5 = nn.Linear(src_emb_dimension, d_model) # concatenated source embedding: ... 
            self.tgt_embed_proj_5 = nn.Linear(tgt_emb_dimension, d_model)
            self.out_embed_proj_5 = nn.Linear(d_model, tgt_emb_dimension)

            # Create the decoder blocks
            decoder_blocks_5 = []
            if v.SHARE_ATTENTION_BLOCKS:
                decoder_self_attention_block_5 = MultiHeadAttentionBlock(d_model, v.HEADS[5], v.DROPOUT_ATT[20])
                decoder_cross_attention_block_5 = MultiHeadAttentionBlock(d_model, v.HEADS[5], v.DROPOUT_ATT[20])
            for _ in range(v.NUM_ATTENTION_BLOCKS[5]):
                if not v.SHARE_ATTENTION_BLOCKS:
                    decoder_self_attention_block_5 = MultiHeadAttentionBlock(d_model, v.HEADS[5], v.DROPOUT_ATT[20])
                    decoder_cross_attention_block_5 = MultiHeadAttentionBlock(d_model, v.HEADS[5], v.DROPOUT_ATT[20])
                feed_forward_block_5 = FeedForwardBlock(d_model, d_model*4, v.DROPOUT_ATT[20])
                decoder_block_5 = DecoderBlock(d_model, decoder_self_attention_block_5, decoder_cross_attention_block_5, feed_forward_block_5, v.DROPOUT_ATT[20])
                decoder_blocks_5.append(decoder_block_5)
            
            # Create the decoder
            self.decoder_5 = Decoder(d_model, nn.ModuleList(decoder_blocks_5))
                
            if v.ADD_POS_ENCOD:
                if not v.SHARE_AUDIO_EMB: self.src_pos_5 = PositionalEnconding(d_model, src_seq_len, 0.0)
                self.tgt_pos_5 = PositionalEnconding(d_model, tgt_seq_len, 0.0)
            else:
                if not v.SHARE_AUDIO_EMB: self.src_pos_5 = None
                self.tgt_pos_5 = None

        
        # Dense Block 6 
        
        self.convT16 = nn.ConvTranspose2d(in_channels = convT15_num_maps, out_channels = convT16_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT16_bn = nn.BatchNorm2d(convT16_num_maps)
        if dropout[21]>0: self.convT16_drop = nn.Dropout2d(0.0)
        
        self.convT17 = nn.ConvTranspose2d(in_channels = convT16_num_maps, out_channels = convT17_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT17_bn = nn.BatchNorm2d(convT17_num_maps)
        if dropout[22]>0: self.convT17_drop = nn.Dropout2d(0.0)
        
        self.convT18 = nn.ConvTranspose2d(in_channels = convT16_num_maps+convT17_num_maps, out_channels = convT18_num_maps, kernel_size = (3,3), stride = 1, padding=1)
        if batchnorm: self.convT18_bn = nn.BatchNorm2d(convT18_num_maps)
        if dropout[23]>0: self.convT18_drop = nn.Dropout2d(0.0)
        
        # Transition 6
        self.transition6 = nn.ConvTranspose2d(in_channels = convT16_num_maps+convT17_num_maps+convT18_num_maps, out_channels = target_channels_num, kernel_size = (4,4), stride = 2, padding=1)
        if batchnorm: self.transition6_bn = nn.BatchNorm2d(target_channels_num)
	
	# Attention Block 6
        if v.NUM_ATTENTION_BLOCKS[6] > 0:
            if v.TRANSPOSE_EMB == (0,0): # Channel Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_seq_len = transition_shape[6][0] * v.PATCHSIZE[6] * v.PATCHSIZE[6]
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_emb_dimension = (transition_shape[6][1] * transition_shape[6][2]) // (v.PATCHSIZE[6] * v.PATCHSIZE[6])
            else: # Spatial Attention
                src_seq_len = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][1] * encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][2]
                tgt_seq_len = (transition_shape[6][1] * transition_shape[6][2]) // (v.PATCHSIZE[6] * v.PATCHSIZE[6])
                src_emb_dimension = encoder_maps_shape[v.AUDIO_ENCODER_LAYER-1][0]
                tgt_emb_dimension = transition_shape[6][0] * v.PATCHSIZE[6] * v.PATCHSIZE[6]

            d_model = v.D_MODEL[6]
            if not v.SHARE_AUDIO_EMB: self.src_embed_proj_6 = nn.Linear(src_emb_dimension, d_model) # concatenated source embedding: ... 
            self.tgt_embed_proj_6 = nn.Linear(tgt_emb_dimension, d_model)
            self.out_embed_proj_6 = nn.Linear(d_model, tgt_emb_dimension)

            # Create the decoder blocks
            decoder_blocks_6 = []
            if v.SHARE_ATTENTION_BLOCKS:
                decoder_self_attention_block_6 = MultiHeadAttentionBlock(d_model, v.HEADS[6], v.DROPOUT_ATT[24])
                decoder_cross_attention_block_6 = MultiHeadAttentionBlock(d_model, v.HEADS[6], v.DROPOUT_ATT[24])
            for _ in range(v.NUM_ATTENTION_BLOCKS[6]):
                if not v.SHARE_ATTENTION_BLOCKS:
                    decoder_self_attention_block_6 = MultiHeadAttentionBlock(d_model, v.HEADS[6], v.DROPOUT_ATT[24])
                    decoder_cross_attention_block_6 = MultiHeadAttentionBlock(d_model, v.HEADS[6], v.DROPOUT_ATT[24])
                feed_forward_block_6 = FeedForwardBlock(d_model, d_model*4, v.DROPOUT_ATT[24])
                decoder_block_6 = DecoderBlock(d_model, decoder_self_attention_block_6, decoder_cross_attention_block_6, feed_forward_block_6, v.DROPOUT_ATT[24])
                decoder_blocks_6.append(decoder_block_6)
            
            # Create the decoder
            self.decoder_6 = Decoder(d_model, nn.ModuleList(decoder_blocks_6))

            if v.ADD_POS_ENCOD:
                if not v.SHARE_AUDIO_EMB: self.src_pos_6 = PositionalEnconding(d_model, src_seq_len, 0.0)
                self.tgt_pos_6 = PositionalEnconding(d_model, tgt_seq_len, 0.0)
            else:
                if not v.SHARE_AUDIO_EMB: self.src_pos_6 = None
                self.tgt_pos_6 = None


                
        
    def forward(self, x):
#         x_size = x.size()[1]*x.size()[2]*x.size()[3]
#         plot = x.view(-1, x_size).cpu().data.numpy()
#         plot_vectors_hist(plot)
        
        # Assign ?-layer output to be used for attention embeddings
        if v.AUDIO_ENCODER_LAYER < 13:
            encoder_out = x[v.AUDIO_ENCODER_LAYER-1] # Only if using encoder inner layers outputs

            # Generate embeddings from encoder last layer output. It will be always used for generator's first layer input
            x_size = x[12].size() # x[12] is the audio encoder 13th (last) layer
            x = x[12].view(x_size[0], x_size[1], x_size[2] * x_size[3]) # Flat two last feature maps dimensions
            
            x = torch.tanh(x) # Apply activation to complete embeddings generation # @UndefinedVariable
        else:
            encoder_out = x.view(-1, self.input_channels_num, self.input_feature_map_shape[0] * self.input_feature_map_shape[1])
            x = x.view(-1, self.input_channels_num, self.input_feature_map_shape[0], self.input_feature_map_shape[1])

        x = self.transition0(x)
        if batchnorm: x = self.transition0_bn(x)
        if dropout[0]>0: x = self.transition0_drop(x)

        # Apply Cross Attention
        if v.NUM_ATTENTION_BLOCKS[0] > 0:
            x_size = x.size()
            src_embed = generate_source_embeddings(encoder_out, v.ADD_POS_ENCOD, self.src_pos_0, v.TRANSPOSE_EMB, self.src_embed_proj_0)
            tgt_embed = generate_target_embeddings(x, v.ADD_POS_ENCOD, self.tgt_pos_0, v.TRANSPOSE_EMB, v.PATCHSIZE[0], self.tgt_embed_proj_0)
            x = apply_attention(src_embed, tgt_embed, self.out_embed_proj_0, v.TRANSPOSE_EMB, x_size, v.PATCHSIZE[0], self.decoder_0).contiguous().view(x_size[0], x_size[1], x_size[2], x_size[3])
        elif (v.SHARE_AUDIO_EMB and v.ATTENTIONAL_NET):
            x_size = x.size()
            src_embed = generate_source_embeddings(encoder_out, v.ADD_POS_ENCOD, self.src_pos_0, v.TRANSPOSE_EMB, self.src_embed_proj_0)
        x = ut.get_activated(x, self.activation[0], alph=self.activation_alpha[0])
        
        x1 = self.convT1(x)
        if batchnorm: x1 = self.convT1_bn(x1)
        if dropout[1]>0: x1 = self.convT1_drop(x1)
        x1 = ut.get_activated(x1, self.activation[1], alph=self.activation_alpha[1])
        
        x2 = self.convT2(x1)
        if batchnorm: x2 = self.convT2_bn(x2)
        if dropout[2]>0: x2 = self.convT2_drop(x2)
        x2 = ut.get_activated(x2, self.activation[2], alph=self.activation_alpha[2])
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable
        
        x3 = self.convT3(x2_dense)
        if batchnorm: x3 = self.convT3_bn(x3)
        if dropout[3]>0: x3 = self.convT3_drop(x3)
        x3 = ut.get_activated(x3, self.activation[3], alph=self.activation_alpha[3])
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable
        
        x = self.transition1(x3_dense)
        if batchnorm: x = self.transition1_bn(x)
        if dropout[4]>0: x = self.transition1_drop(x)
        
        # Apply Cross Attention
        if v.NUM_ATTENTION_BLOCKS[1] > 0:
            x_size = x.size()
            if not v.SHARE_AUDIO_EMB:
                src_embed = generate_source_embeddings(encoder_out, v.ADD_POS_ENCOD, self.src_pos_1, v.TRANSPOSE_EMB, self.src_embed_proj_1)
            tgt_embed = generate_target_embeddings(x, v.ADD_POS_ENCOD, self.tgt_pos_1, v.TRANSPOSE_EMB, v.PATCHSIZE[1], self.tgt_embed_proj_0)
            x = apply_attention(src_embed, tgt_embed, self.out_embed_proj_0, v.TRANSPOSE_EMB, x_size, v.PATCHSIZE[1], self.decoder_0).contiguous().view(x_size[0], x_size[1], x_size[2], x_size[3])

        x = ut.get_activated(x, self.activation[4], alph=self.activation_alpha[4])
        
        x1 = self.convT4(x)
        if batchnorm: x1 = self.convT4_bn(x1)
        if dropout[5]>0: x1 = self.convT4_drop(x1)
        x1 = ut.get_activated(x1, self.activation[5], alph=self.activation_alpha[5])
        
        x2 = self.convT5(x1)
        if batchnorm: x2 = self.convT5_bn(x2)
        if dropout[6]>0: x2 = self.convT5_drop(x2)
        x2 = ut.get_activated(x2, self.activation[6], alph=self.activation_alpha[6])
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable
        
        x3 = self.convT6(x2_dense)
        if batchnorm: x3 = self.convT6_bn(x3)
        if dropout[7]>0: x3 = self.convT6_drop(x3)
        x3 = ut.get_activated(x3, self.activation[7], alph=self.activation_alpha[7])
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable
        
        x = self.transition2(x3_dense)
        if batchnorm: x = self.transition2_bn(x)
        if dropout[8]>0: x = self.transition2_drop(x)
        
        # Apply Cross Attention
        if v.NUM_ATTENTION_BLOCKS[2] > 0:
            x_size = x.size()
            if not v.SHARE_AUDIO_EMB:
                src_embed = generate_source_embeddings(encoder_out, v.ADD_POS_ENCOD, self.src_pos_2, v.TRANSPOSE_EMB, self.src_embed_proj_2)
            tgt_embed = generate_target_embeddings(x, v.ADD_POS_ENCOD, self.tgt_pos_2, v.TRANSPOSE_EMB, v.PATCHSIZE[2], self.tgt_embed_proj_2)
            x = apply_attention(src_embed, tgt_embed, self.out_embed_proj_2, v.TRANSPOSE_EMB, x_size, v.PATCHSIZE[2], self.decoder_2).contiguous().view(x_size[0], x_size[1], x_size[2], x_size[3])

        x = ut.get_activated(x, self.activation[8], alph=self.activation_alpha[8])
        
        x1 = self.convT7(x)
        if batchnorm: x1 = self.convT7_bn(x1)
        if dropout[9]>0: x1 = self.convT7_drop(x1)
        x1 = ut.get_activated(x1, self.activation[9], alph=self.activation_alpha[9])
        
        x2 = self.convT8(x1)
        if batchnorm: x2 = self.convT8_bn(x2)
        if dropout[10]>0: x2 = self.convT8_drop(x2)
        x2 = ut.get_activated(x2, self.activation[10], alph=self.activation_alpha[10])
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable
        
        x3 = self.convT9(x2_dense)
        if batchnorm: x3 = self.convT9_bn(x3)
        if dropout[11]>0: x3 = self.convT9_drop(x3)
        x3 = ut.get_activated(x3, self.activation[11], alph=self.activation_alpha[11])
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable
        
        x = self.transition3(x3_dense)
        if batchnorm: x = self.transition3_bn(x)
        if dropout[12]>0: x = self.transition3_drop(x)

        # Apply Cross Attention
        if v.NUM_ATTENTION_BLOCKS[3] > 0:
            x_size = x.size()
            if not v.SHARE_AUDIO_EMB:
                src_embed = generate_source_embeddings(encoder_out, v.ADD_POS_ENCOD, self.src_pos_3, v.TRANSPOSE_EMB, self.src_embed_proj_3)
            tgt_embed = generate_target_embeddings(x, v.ADD_POS_ENCOD, self.tgt_pos_3, v.TRANSPOSE_EMB, v.PATCHSIZE[3], self.tgt_embed_proj_3)
            x = apply_attention(src_embed, tgt_embed, self.out_embed_proj_3, v.TRANSPOSE_EMB, x_size, v.PATCHSIZE[3], self.decoder_3).contiguous().view(x_size[0], x_size[1], x_size[2], x_size[3])
        
        x = ut.get_activated(x, self.activation[12], alph=self.activation_alpha[12])
        
        x1 = self.convT10(x)
        if batchnorm: x1 = self.convT10_bn(x1)
        if dropout[13]>0: x1 = self.convT10_drop(x1)
        x1 = ut.get_activated(x1, self.activation[13], alph=self.activation_alpha[13])
        
        x2 = self.convT11(x1)
        if batchnorm: x2 = self.convT11_bn(x2)
        if dropout[14]>0: x2 = self.convT11_drop(x2)
        x2 = ut.get_activated(x2, self.activation[14], alph=self.activation_alpha[14])
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable
        
        x3 = self.convT12(x2_dense)
        if batchnorm: x3 = self.convT12_bn(x3)
        if dropout[15]>0: x3 = self.convT12_drop(x3)
        x3 = ut.get_activated(x3, self.activation[15], alph=self.activation_alpha[15])
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable
        
        x = self.transition4(x3_dense)
        if batchnorm: x = self.transition4_bn(x)
        if dropout[16]>0: x = self.transition4_drop(x)
        
        # Apply Cross Attention
        if v.NUM_ATTENTION_BLOCKS[4] > 0:
            x_size = x.size()
            if not v.SHARE_AUDIO_EMB:
                src_embed = generate_source_embeddings(encoder_out, v.ADD_POS_ENCOD, self.src_pos_4, v.TRANSPOSE_EMB, self.src_embed_proj_4)
            tgt_embed = generate_target_embeddings(x, v.ADD_POS_ENCOD, self.tgt_pos_4, v.TRANSPOSE_EMB, v.PATCHSIZE[4], self.tgt_embed_proj_4)
            x = apply_attention(src_embed, tgt_embed, self.out_embed_proj_4, v.TRANSPOSE_EMB, x_size, v.PATCHSIZE[4], self.decoder_4).contiguous().view(x_size[0], x_size[1], x_size[2], x_size[3])

        x = ut.get_activated(x, self.activation[16], alph=self.activation_alpha[16])
        
        x1 = self.convT13(x)
        if batchnorm: x1 = self.convT13_bn(x1)
        if dropout[17]>0: x1 = self.convT13_drop(x1)
        x1 = ut.get_activated(x1, self.activation[17], alph=self.activation_alpha[17])
        
        x2 = self.convT14(x1)
        if batchnorm: x2 = self.convT14_bn(x2)
        if dropout[18]>0: x2 = self.convT14_drop(x2)
        x2 = ut.get_activated(x2, self.activation[18], alph=self.activation_alpha[18])
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable
        
        x3 = self.convT15(x2_dense)
        if batchnorm: x3 = self.convT15_bn(x3)
        if dropout[19]>0: x3 = self.convT15_drop(x3)
        x3 = ut.get_activated(x3, self.activation[19], alph=self.activation_alpha[19])
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable
        
        x = self.transition5(x3_dense)
        if batchnorm: x = self.transition5_bn(x)
        if dropout[20]>0: x = self.transition5_drop(x)
        
        # Apply Cross Attention
        if v.NUM_ATTENTION_BLOCKS[5] > 0:
            x_size = x.size()
            if not v.SHARE_AUDIO_EMB:
                src_embed = generate_source_embeddings(encoder_out, v.ADD_POS_ENCOD, self.src_pos_5, v.TRANSPOSE_EMB, self.src_embed_proj_5)
            tgt_embed = generate_target_embeddings(x, v.ADD_POS_ENCOD, self.tgt_pos_5, v.TRANSPOSE_EMB, v.PATCHSIZE[5], self.tgt_embed_proj_5)
            x = apply_attention(src_embed, tgt_embed, self.out_embed_proj_5, v.TRANSPOSE_EMB, x_size, v.PATCHSIZE[5], self.decoder_5).contiguous().view(x_size[0], x_size[1], x_size[2], x_size[3])

        x = ut.get_activated(x, self.activation[20], alph=self.activation_alpha[20])
        
        x1 = self.convT16(x)
        if batchnorm: x1 = self.convT16_bn(x1)
        if dropout[21]>0: x1 = self.convT16_drop(x1)
        x1 = ut.get_activated(x1, self.activation[21], alph=self.activation_alpha[21])
        
        x2 = self.convT17(x1)
        if batchnorm: x2 = self.convT17_bn(x2)
        if dropout[22]>0: x2 = self.convT17_drop(x2)
        x2 = ut.get_activated(x2, self.activation[22], alph=self.activation_alpha[22])
        x2_dense = torch.cat((x1, x2), 1)  # @UndefinedVariable
        
        x3 = self.convT18(x2_dense)
        if batchnorm: x3 = self.convT18_bn(x3)
        if dropout[23]>0: x3 = self.convT18_drop(x3)
        x3 = ut.get_activated(x3, self.activation[23], alph=self.activation_alpha[23])
        x3_dense = torch.cat((x1, x2, x3), 1)  # @UndefinedVariable
        
        x = self.transition6(x3_dense)

        if batchnorm: x = self.transition6_bn(x)
        
        # Apply Cross Attention
        if v.NUM_ATTENTION_BLOCKS[6] > 0:
            x_size = x.size()
            if not v.SHARE_AUDIO_EMB:
                src_embed = generate_source_embeddings(encoder_out, v.ADD_POS_ENCOD, self.src_pos_6, v.TRANSPOSE_EMB, self.src_embed_proj_6)
            tgt_embed = generate_target_embeddings(x, v.ADD_POS_ENCOD, self.tgt_pos_6, v.TRANSPOSE_EMB, v.PATCHSIZE[6], self.tgt_embed_proj_6)
            x = apply_attention(src_embed, tgt_embed, self.out_embed_proj_6, v.TRANSPOSE_EMB, x_size, v.PATCHSIZE[6], self.decoder_6).contiguous().view(x_size[0], x_size[1], x_size[2], x_size[3])
                
        
        x = torch.tanh(x)  # @UndefinedVariable
        
        return x
    
    
    def summary(self):
        for param in self.parameters():
            #print(type(param.data), param.size())
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
    # Must always use at least the transforms.ToTensor() to avoid passing an unprocessed PIL image to the 
    # DataLoader, whereas it should receive a tensor.
    if v.AUDIO_DATA_AUGMENTATION:
        # The input data loaded are spectrograms that will need to be encoded before being fed to the net
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomApply((transforms.RandomResizedCrop((input_height,input_width),scale=(0.95, 1.0),ratio=(0.95, 1.0)),), p=0.9), # scale=(0.9, 1.0),ratio=(1.0, 1.0)),), p=1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif v.TRAIN_AUDIO_ENCODER | (v.AUDIO_ENCODER_LAYER < 13):
        # The input data loaded are spectrograms that will need to be encoded before being fed to the net
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else: 
        transform = None # No transformation because it is the embedding that is being loaded
    
    if v.VISUAL_DATA_AUGMENTATION:
        target_transform = transforms.Compose([
            transforms.RandomApply((transforms.RandomResizedCrop((target_height,target_width),scale=(0.75, 1.0),ratio=(0.9, 1.0)),), p=0.9), # scale=(0.75, 1.0),ratio=(1.0, 1.0)),), p=1.0),
            transforms.RandomApply((transforms.RandomHorizontalFlip(),), p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        target_transform = transforms.Compose([
            #transforms.Resize((227, 227)), #for AlexNet
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    if v.AUDIO_DATA_AUGMENTATION | v.TRAIN_AUDIO_ENCODER | (v.AUDIO_ENCODER_LAYER < 13):
        # The input data loaded are spectrograms that will need to be encoded before being fed to the net
        trainset = vegas_visual_generator_s2i.VEGAS_VISUAL_GENERATOR_S2I(
            root=v.BATCHES_AUDIO_VISUAL_SPECTROGRAM_TO_IMAGE_DIR, label_length=label_length, input_height=input_height, input_width=input_width, 
            target_height=target_height, target_width=target_width, target_channels_num=target_channels_num, train=True, transform=transform, 
            target_transform=target_transform)  # @UndefinedVariable
    else:
        trainset = vegas_visual_generator.VEGAS_VISUAL_GENERATOR(
            root=v.BATCHES_AUDIO_VISUAL_EMBEDDING_TO_IMAGE_DIR, label_length=label_length, input_length=audio_embedding_dim, target_height=target_height, 
            target_width=target_width, target_channels_num=target_channels_num, train=True, transform=transform, target_transform=target_transform)  # @UndefinedVariable
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=shuffle, num_workers=2)  # @UndefinedVariable
    return trainloader

def get_test_loader(shuffle=False):
    
    if v.AUDIO_DATA_AUGMENTATION | v.TRAIN_AUDIO_ENCODER | (v.AUDIO_ENCODER_LAYER < 13):
        # The input data loaded are spectrograms that will need to be encoded before being fed to the net
        transform_test = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform_test = None # No transformation because it is the embedding that is being loaded
    
    target_transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if v.AUDIO_DATA_AUGMENTATION | v.TRAIN_AUDIO_ENCODER | (v.AUDIO_ENCODER_LAYER < 13):
        # The input data loaded are spectrograms that will need to be encoded before being fed to the net
        testset = vegas_visual_generator_s2i.VEGAS_VISUAL_GENERATOR_S2I(
            root=v.BATCHES_AUDIO_VISUAL_SPECTROGRAM_TO_IMAGE_DIR, label_length=label_length, input_height=input_height, input_width=input_width, 
            target_height=target_height, target_width=target_width, target_channels_num=target_channels_num, train=False, transform=transform_test, 
            target_transform=target_transform_test)  # @UndefinedVariable
    else:
        testset = vegas_visual_generator.VEGAS_VISUAL_GENERATOR(
            root=v.BATCHES_AUDIO_VISUAL_EMBEDDING_TO_IMAGE_DIR, label_length=label_length, input_length=audio_embedding_dim, target_height=target_height, 
            target_width=target_width, target_channels_num=target_channels_num, train=False, transform=transform_test, target_transform=target_transform_test)  # @UndefinedVariable
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=shuffle, num_workers=2)  # @UndefinedVariable
    return testloader  

def plot_vectors_hist(vectors):
    import matplotlib.pyplot as plt
    i = 0
    v_list = []
    colors = []
    sum_vector_mean = 0
    sum_vector_std = 0
    for v in vectors:
        v_list.append(v)
        colors.append('b')
        sum_vector_mean += np.mean(v)
        sum_vector_std += np.std(v, ddof=1)
        i += 1
    vector_mean = sum_vector_mean / i
    vector_std = sum_vector_std / i
    count, bins, ignored = plt.hist(v_list, 30, density=False, color=colors)
    #if density: plt.plot(bins, 1/(vector_std * np.sqrt(2 * np.pi)) * np.exp( - (bins - vector_mean)**2 / (2 * vector_std**2) ),linewidth=2, color='r')
    plt.hist(v_list, 30, density=False, color=colors)
#     plt.ylim(ymin=(0,300))
#     plt.xlim(xmin=(-0.2,0.2))
    plt.show()

def get_dimensions (input_embedding_dim):
    if input_embedding_dim == 128:
        input_feature_map_shape = (1,1)
        input_feature_map_flat_dim = input_feature_map_shape[0] * input_feature_map_shape[1]
        if v.OVERLAP_NOISE:
            input_channels_num = 128
            convT1_num_maps = 512 
            convT2_num_maps = 512
            convT3_num_maps = 512
        else:
            input_channels_num = 128 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT1_num_maps = 512 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT2_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)*2/3)
            convT3_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)/3)
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
        convT1_to_17_shape = ((512,2,2),(512,2,2),(512,2,2),(512,4,4),(416,4,4),(352,4,4),(304,6,6),(256,6,6),(224,6,6),(192,12,12),(176,12,12),(160,12,12),(144,24,24),(128,24,24),(112,24,24),(96,48,48),(80,48,48))
        transition_shape = ((512,2,2),(512,4,4),(352,6,6),(224,12,12),(160,24,24),(112,48,48),(3,96,96))
        encoder_maps_shape = ((80,61,48),(176,29,23),(192,25,21),(224,21,19),(272,17,17),(304,15,15),(352,13,13),(416,11,11),(512,9,9),(512,7,7),(512,5,5),(512,3,3),(512,1,1)) # Folows net_audio_autoencoder architecture
    elif input_embedding_dim == 192:
        input_feature_map_shape = (1,1)
        input_feature_map_flat_dim = input_feature_map_shape[0] * input_feature_map_shape[1]
        if v.OVERLAP_NOISE:
            input_channels_num = 192
            convT1_num_maps = 512 
            convT2_num_maps = 512
            convT3_num_maps = 512
        else:
            input_channels_num = 192 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT1_num_maps = 512 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT2_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)*2/3)
            convT3_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)/3)
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
        convT1_to_17_shape = ((512,2,2),(512,2,2),(512,2,2),(512,4,4),(416,4,4),(352,4,4),(304,6,6),(256,6,6),(224,6,6),(192,12,12),(176,12,12),(160,12,12),(144,24,24),(128,24,24),(112,24,24),(96,48,48),(80,48,48))
        transition_shape = ((512,2,2),(512,4,4),(352,6,6),(224,12,12),(160,24,24),(112,48,48),(3,96,96))
        encoder_maps_shape = ((80,61,48),(176,29,23),(192,25,21),(224,21,19),(272,17,17),(304,15,15),(352,13,13),(416,11,11),(512,9,9),(512,7,7),(512,5,5),(512,3,3),(512,1,1)) # Folows net_audio_autoencoder architecture
    elif input_embedding_dim == 256:
        input_feature_map_shape = (1,1)
        input_feature_map_flat_dim = input_feature_map_shape[0] * input_feature_map_shape[1]
        if v.OVERLAP_NOISE:
            input_channels_num = 256
            convT1_num_maps = 512 
            convT2_num_maps = 512
            convT3_num_maps = 512
        else:
            input_channels_num = 256 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT1_num_maps = 512 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT2_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)*2/3)
            convT3_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)/3)
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
        convT1_to_17_shape = ((512,2,2),(512,2,2),(512,2,2),(512,4,4),(416,4,4),(352,4,4),(304,6,6),(256,6,6),(224,6,6),(192,12,12),(176,12,12),(160,12,12),(144,24,24),(128,24,24),(112,24,24),(96,48,48),(80,48,48))
        transition_shape = ((512,2,2),(512,4,4),(352,6,6),(224,12,12),(160,24,24),(112,48,48),(3,96,96))
        encoder_maps_shape = ((80,61,48),(176,29,23),(192,25,21),(224,21,19),(272,17,17),(304,15,15),(352,13,13),(416,11,11),(512,9,9),(512,7,7),(512,5,5),(512,3,3),(512,1,1)) # Folows net_audio_autoencoder architecture
    elif input_embedding_dim == 512:
        input_feature_map_shape = (1,1)
        input_feature_map_flat_dim = input_feature_map_shape[0] * input_feature_map_shape[1]
        if v.OVERLAP_NOISE:
            input_channels_num = 512
            convT1_num_maps = 512 
            convT2_num_maps = 512
            convT3_num_maps = 512
        else:
            input_channels_num = 512 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT1_num_maps = 512 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT2_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)*2/3)
            convT3_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)/3)
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
        convT1_to_17_shape = ((512,2,2),(512,2,2),(512,2,2),(512,4,4),(416,4,4),(352,4,4),(304,6,6),(256,6,6),(224,6,6),(192,12,12),(176,12,12),(160,12,12),(144,24,24),(128,24,24),(112,24,24),(96,48,48),(80,48,48))
        transition_shape = ((512,2,2),(512,4,4),(352,6,6),(224,12,12),(160,24,24),(112,48,48),(3,96,96))
        encoder_maps_shape = ((80,61,48),(176,29,23),(192,25,21),(224,21,19),(272,17,17),(304,15,15),(352,13,13),(416,11,11),(512,9,9),(512,7,7),(512,5,5),(512,3,3),(512,1,1)) # Folows net_audio_autoencoder architecture
    elif input_embedding_dim == 1024:
        input_feature_map_shape = (1,1)
        input_feature_map_flat_dim = input_feature_map_shape[0] * input_feature_map_shape[1]
        if v.OVERLAP_NOISE:
            input_channels_num = 1024
            convT1_num_maps = 512 
            convT2_num_maps = 512
            convT3_num_maps = 512
        else:
            input_channels_num = 1024 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT1_num_maps = 512 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT2_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)*2/3)
            convT3_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)/3)
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
        convT1_to_17_shape = ((512,2,2),(512,2,2),(512,2,2),(512,4,4),(416,4,4),(352,4,4),(304,6,6),(256,6,6),(224,6,6),(192,12,12),(176,12,12),(160,12,12),(144,24,24),(128,24,24),(112,24,24),(96,48,48),(80,48,48))
        transition_shape = ((512,2,2),(512,4,4),(352,6,6),(224,12,12),(160,24,24),(112,48,48),(3,96,96))
        encoder_maps_shape = ((80,61,48),(176,29,23),(192,25,21),(224,21,19),(272,17,17),(304,15,15),(352,13,13),(416,11,11),(512,9,9),(512,7,7),(512,5,5),(512,3,3),(512,1,1)) # Folows net_audio_autoencoder architecture
    elif input_embedding_dim == 2048:
        input_feature_map_shape = (1,1)
        input_feature_map_flat_dim = input_feature_map_shape[0] * input_feature_map_shape[1]
        if v.OVERLAP_NOISE:
            input_channels_num = 2048
            convT1_num_maps = 512 
            convT2_num_maps = 512
            convT3_num_maps = 512
        else:
            input_channels_num = 2048 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT1_num_maps = 512 + int(noise_vector_dim/input_feature_map_flat_dim)
            convT2_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)*2/3)
            convT3_num_maps = 512 + int((noise_vector_dim/input_feature_map_flat_dim)/3)
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
        convT1_to_17_shape = ((512,2,2),(512,2,2),(512,2,2),(512,4,4),(416,4,4),(352,4,4),(304,6,6),(256,6,6),(224,6,6),(192,12,12),(176,12,12),(160,12,12),(144,24,24),(128,24,24),(112,24,24),(96,48,48),(80,48,48))
        transition_shape = ((512,2,2),(512,4,4),(352,6,6),(224,12,12),(160,24,24),(112,48,48),(3,96,96))
        encoder_maps_shape = ((80,61,48),(176,29,23),(192,25,21),(224,21,19),(272,17,17),(304,15,15),(352,13,13),(416,11,11),(512,9,9),(512,7,7),(512,5,5),(512,3,3),(512,1,1)) # Folows net_audio_autoencoder architecture
    
    return (input_feature_map_shape,input_channels_num,convT1_num_maps,convT2_num_maps,convT3_num_maps,convT4_num_maps,convT5_num_maps,convT6_num_maps,
            convT7_num_maps,convT8_num_maps,convT9_num_maps,convT10_num_maps,convT11_num_maps,convT12_num_maps,convT13_num_maps,convT14_num_maps,
            convT15_num_maps,convT16_num_maps,convT17_num_maps,convT18_num_maps,convT1_to_17_shape,transition_shape,encoder_maps_shape)

# Cloned from https://github.com/hkproj/pytorch-transformer
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Cloned from https://github.com/hkproj/pytorch-transformer
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Adapted from https://github.com/hkproj/pytorch-transformer
class PositionalEnconding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len    
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_mopdel)
        pe = torch.zeros(seq_len, d_model) # @UndefinedVariable
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1) # (seq_len, 1) # @UndefinedVariable
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2) # @UndefinedVariable
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model)) # @UndefinedVariable
        # Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model)) # @UndefinedVariable
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

# Adapted from https://github.com/hkproj/pytorch-transformer
class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            if (self.dropout.p > 0.0):
                return self.dropout(self.norm(x + sublayer(x)))
            else:
                return self.norm(x + sublayer(x))

# Adapted from https://github.com/hkproj/pytorch-transformer
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by x"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout, training: bool=False):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if (dropout.p > 0.0) and training:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
        
    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        
        # Split embedding to create smaller matrices for using with multiple heads
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout, self.training)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

# Cloned from https://github.com/hkproj/pytorch-transformer
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

# Cloned from https://github.com/hkproj/pytorch-transformer
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # return self.norm(x)
        return x # Do not apply the normalization since now it is being applied at the end of each residual connection


# Create embeddings sequence
def generate_source_embeddings(encoder_output, add_pos_encod, src_pos_encod, transpose, src_embed_proj=None) -> None:
    encoder_out_size = encoder_output.size()
    # if v.AUDIO_DATA_AUGMENTATION | v.TRAIN_AUDIO_ENCODER | (v.AUDIO_ENCODER_LAYER < 13): # Only in these cases, otherwise the embeddings are already activated 
    #     encoder_output = torch.tanh(encoder_output)
    if transpose[0] == 1: encoder_output = encoder_output.transpose(1,2)
    encoder_output = src_embed_proj(encoder_output)
    #encoder_output = torch.tanh(encoder_output)  # @UndefinedVariable

    if add_pos_encod:
        encoder_output = src_pos_encod(encoder_output)

    return encoder_output

# Create embeddings sequence
def generate_target_embeddings(x, add_pos_encod, tgt_pos_encod, transpose, patch_size, tgt_embed_proj=None) -> None:
    x_size = x.size()
    x = torch.tanh(x) # Activate before linear projection to be more coherent with audio embeddings that are forwarded in the generator 1st layer
    if patch_size == 1:
        tgt_embed = x.view(x_size[0], x_size[1], x_size[2] * x_size[3])
        if transpose[0] == 1: tgt_embed = tgt_embed.transpose(1,2)
    else:
        # Patchify
        unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        tgt_embed = unfold(x)
        tgt_embed = tgt_embed.view(x_size[0], x_size[1],patch_size, patch_size, -1)
        if transpose[0] == 1: tgt_embed = tgt_embed.permute(0, 4, 1, 2, 3)
        tgt_embed_size = tgt_embed.size()
        tgt_embed = tgt_embed.view(tgt_embed_size[0], tgt_embed_size[1], tgt_embed_size[2], -1) # Flatten last 2 dimensions that corresponds to the patches
        tgt_embed = tgt_embed.view(tgt_embed_size[0], tgt_embed_size[1], -1)  # Flatten together patches and channels
        # if transpose[0] == 0: tgt_embed = tgt_embed.transpose(1,2) # Here for channel attention we need to transpose
    tgt_embed = tgt_embed_proj(tgt_embed)
    #tgt_embed = torch.tanh(tgt_embed)  # @UndefinedVariable

    if add_pos_encod:
        tgt_embed = tgt_pos_encod(tgt_embed)

    return tgt_embed


def apply_attention(encoder_output, tgt_embed, out_embed_proj, transpose, feature_map_size, patch_size, decoder: Decoder) -> None:

    attention_res = out_embed_proj(decoder(tgt_embed, encoder_output, None, None))

    if patch_size == 1:
        if transpose[1] == 1: attention_res = attention_res.transpose(1,2)
    else:
        # Unpatchify
        attention_res = attention_res.view(attention_res.size()[0], attention_res.size()[1], -1, patch_size * patch_size)
        attention_res = attention_res.view(attention_res.size()[0], attention_res.size()[1], attention_res.size()[2], patch_size, patch_size)
        attention_res_size = attention_res.size()
        attention_res = attention_res.permute(0, 2, 3, 4, 1).view(attention_res_size[0], -1, attention_res_size[1])
        fold = torch.nn.Fold(output_size=(feature_map_size[2],feature_map_size[3]), kernel_size=patch_size, stride=patch_size)
        attention_res = fold(attention_res)
        attention_res = attention_res.view(attention_res.size()[0], attention_res.size()[1], attention_res.size()[2] * attention_res.size()[3]) # Will be unflattened after

    return attention_res

