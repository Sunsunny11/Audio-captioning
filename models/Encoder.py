#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

"""
Adapt from Qiuqiang Kong's code: https://github.com/qiuqiangkong/audioset_tagging_cnn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from models.SpecAugment import SpecAugmentation


class Pooling_layer(nn.Module):
    def __init__(self, pooling_type='avg-max', factor=0.5):

        super(Pooling_layer, self).__init__()
        self.factor = factor
        self.pooling_type = pooling_type

    def forward(self, x):

        """
        args:
            x: input mel spectrogram [batch, 1, time, frequency]
        return:
            out: reduced features [batch, 1, factor, frequency]
        """

        factor = int(x.shape[2] * self.factor)
        if self.pooling_type == 'avg':
            size = x.shape[2] // factor
            out = F.avg_pool2d(x, kernel_size=(size, 1))
        elif self.pooling_type == 'max':
            size = x.shape[2] // factor
            out = F.max_pool2d(x, kernel_size=(size, 1))
        elif self.pooling_type == 'avg-max':
            size = x.shape[2] // factor
            out1 = F.max_pool2d(x, kernel_size=(size, 1))
            out2 = F.avg_pool2d(x, kernel_size=(size, 1))
            out = out1 + out2
        return out


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.max_pool2d(x, kernel_size=pool_size)
            x2 = F.avg_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn10(nn.Module):

    def __init__(self, config):

        super(Cnn10, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        sr = config.wav.sr
        window_size = config.wav.window_size
        hop_length = config.wav.hop_length
        mel_bins = config.wav.mel_bins
        fmin = config.wav.fmin
        fmax = config.wav.fmax

        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_length,
                                                 win_length=window_size,
                                                 window='hann',
                                                 center=True,
                                                 pad_mode='reflect',
                                                 freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sr, n_fft=window_size,
                                                 n_mels=mel_bins,
                                                 fmin=fmin,
                                                 fmax=fmax,
                                                 ref=1.0,
                                                 amin=1e-10,
                                                 top_db=None,
                                                 freeze_parameters=True)

        self.is_spec_augment = config.training.spec_augmentation

        if self.is_spec_augment:
            self.spec_augmenter = SpecAugmentation(time_drop_width=64,
                                                   time_stripes_num=2,
                                                   freq_drop_width=8,
                                                   freq_stripes_num=2,
                                                   mask_type='zero_value')

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 1024, bias=True)
        #self.fc2 = nn.Linear(256, 128, bias=True)  #x3 for original
        self.fc2 = nn.Linear(128, 128, bias=True)

        self.init_weights()

        self.pooling_layer = Pooling_layer()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, input, pooling=True):
        """ input: (batch_size, time_steps, mel_bins)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.is_spec_augment:
            x = self.spec_augmenter(x)

        if pooling:
            x = self.pooling_layer(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x1 = F.dropout(x, p=0.2, training=self.training)               #128(32,128,322,16) #LHDF compared experience
        x = self.conv_block3(x1, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)                      ##LHDFF original uses the third layer
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)   #average in the frequency domain (batch_size, channel, time)(32,512,80)
        x2 = torch.mean(x1, dim=3) #(32,128,322)
        x3 = x2.permute(2, 0, 1)   #time x batch x channel (322, 32, 128)
        x3 = F.relu_(self.fc2(x3))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = x.permute(2, 0, 1)  # time x batch x channel (512)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        # x = F.relu_(self.fc2(x))
        # x = F.dropout(x, p=0.2, training=self.training)
        src_output = {'x3': x3, 'x': x}
        return src_output


class Cnn14(nn.Module):

    def __init__(self, config):

        super(Cnn14, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        sr = config.wave.sr
        window_size = config.wave.window_size
        hop_length = config.wave.hop_length
        mel_bins = config.wave.mel_bins
        fmin = config.wave.fmin
        fmax = config.wave.fmax

        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_length,
                                                 win_length=window_size,
                                                 window='hann',
                                                 center=True,
                                                 pad_mode='reflect',
                                                 freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sr, n_fft=window_size,
                                                 n_mels=mel_bins,
                                                 fmin=fmin,
                                                 fmax=fmax,
                                                 ref=1.0,
                                                 amin=1e-10,
                                                 top_db=None,
                                                 freeze_parameters=True)

        self.is_spec_augment = config.training.spec_augmentation

        if self.is_spec_augment:
            self.spec_augmenter = SpecAugmentation(time_drop_width=64,
                                                   time_stripes_num=2,
                                                   freq_drop_width=8,
                                                   freq_stripes_num=2,
                                                   mask_type='zero_value')

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 1024, bias=True)

        self.init_weights()

    def init_weights(self):

        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input):
        """ input: (batch_size, time_steps, mel_bins)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.is_spec_augment:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)  # average in the frequency domain (batch_size, channel, time)

        x = x.permute(2, 0, 1)  # time x batch x channel (2048)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)

        return x
