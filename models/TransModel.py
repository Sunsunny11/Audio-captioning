#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @ CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer,\
TransformerDecoder, TransformerDecoderLayer
from models.Encoder import Cnn10, Cnn14
from tools.file_io import load_pickle_file
from tools.utils import align_word_embedding
from models.net_vlad import NetVLAD

from tools.AudioTransformer import AudioTransformer

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).

    """

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional audio_encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """ Container module with an Cnn audio_encoder and a Transformer decoder."""

    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.model_type = 'Cnn+Transformer'

        vocabulary = load_pickle_file(config.path.vocabulary.format(config.dataset))
        ntoken = len(vocabulary)

        # setting for CNN
        if config.encoder.model == 'Cnn10':
            self.feature_extractor = Cnn10(config)
        elif config.encoder.model == 'Cnn14':
            self.feature_extractor = Cnn14(config)
        else:
            raise NameError('No such enocder model')

        if config.encoder.pretrained:
            pretrained_cnn = torch.load('pretrained_models/audio_encoder/{}.pth'.
                                        format(config.encoder.model))['model']
            dict_new = self.feature_extractor.state_dict().copy()
            trained_list = [i for i in pretrained_cnn.keys()
                            if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
            for i in range(len(trained_list)):
                dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
            self.feature_extractor.load_state_dict(dict_new)
        if config.encoder.freeze:
            for name, p in self.feature_extractor.named_parameters():
                p.requires_grad = False

        # decoder settings
        self.decoder_only = config.decoder.decoder_only
        nhead = config.decoder.nhead       # number of heads in Transformer
        self.nhid = config.decoder.nhid         # number of expected features in decoder inputs
        nlayers = config.decoder.nlayers   # number of sub-decoder-layer in the decoder
        dim_feedforward = config.decoder.dim_feedforward   # dimension of the feedforward model
        activation = config.decoder.activation     # activation function of decoder intermediate layer
        dropout = config.decoder.dropout   # the dropout value

        self.pos_encoder = PositionalEncoding(self.nhid, dropout)

        if not self.decoder_only:
            ''' Including transfomer audio_encoder '''
            encoder_layers = TransformerEncoderLayer(self.nhid,
                                                     nhead,
                                                     dim_feedforward,
                                                     dropout,
                                                     activation)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        decoder_layers = TransformerDecoderLayer(self.nhid,
                                                 nhead,
                                                 dim_feedforward,
                                                 dropout,
                                                 activation)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        # linear layers
        self.audio_linear = nn.Linear(1024, self.nhid, bias=True)
        self.dec_fc = nn.Linear(self.nhid, ntoken)
        self.generator = nn.Softmax(dim=-1)
        self.word_emb = nn.Embedding(ntoken, self.nhid)

        self.is_vlad = config.training.vlad
        if self.is_vlad:
            self.net_vlad = NetVLAD(cluster_size=20, feature_size=128)   #32  20

        # attention
        #self.is_AudioTransformer = config.training.AudioTransformer
        #if self.is_AudioTransformer:
            #self.AudioTransformer = AudioTransformer(dim=80, depth=1, heads=2, mlp_dim=1024, dim_head=80, dropout=0.1)   # dim=128
        #self.AudioTransformer = AudioTransformer(dim=128, depth=1, heads=2, mlp_dim=1024, dim_head=128,
                                                 #dropout=0.1)        #dim = dim_head = 20 here, dim=noncomputeattention_matrix
        self.init_weights()

        # setting for pretrained word embedding
        if config.word_embedding.freeze:
            self.word_emb.weight.requires_grad = False
        if config.word_embedding.pretrained:
            self.word_emb.weight.data = align_word_embedding(config.path.vocabulary.format(config.dataset),
                                                             config.path.word2vec, config.decoder.nhid)

    def init_weights(self):
        # initrange = 0.1
        # self.word_emb.weight.data.uniform_(-initrange, initrange)
        init_layer(self.audio_linear)
        # init_layer(self.dec_fc)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src):        #for the original
        #src_output = {'x3': x3, 'x': x}    x3: time x batch x channel (322, 32, 128)  
        src_output = self.feature_extractor(src)  # (time, batch, feature)
        src = F.relu_(self.audio_linear(src_output['x']))
        
        src = F.dropout(src, p=0.2, training=self.training)    #(time, batch, feature)
        src_layer2 = src_output['x3']     #(time, batch, feature)
        
        src0 = src.transpose(1, 0)
        src1 = src0.transpose(2, 1)
        
        srclen = src.shape[0]
        srclen2 = src_layer2.shape[0]
        
        src3 = F.pad(src1, (srclen2 - srclen, 0))  
        src4 = src3.transpose(2, 1) 
        src5 = src4.transpose(1, 0)    # (time,batch,feature)
        src6 = src_layer2 + src5
        
        ############################# vlad
        ###if self.is_vlad:
            # src = src.transpose(1, 0)    #32 80 128
            # src2 = self.net_vlad(src)      # batchsize * T * dimension   32 20 128
            ###src1 = src.transpose(1, 0)  # 32 80 128
            ###src2 = self.net_vlad(src1)  # batchsize * T * dimension   32 20 128
            ###src_vlad = src2.transpose(1, 0)
            ###src3 = src2.transpose(2, 1)
            ###srclen = src1.shape[1]
            ###src2len = src2.shape[1]
            ###src4 = F.pad(src3, (srclen - src2len, 0))  # batchsize * T * dimension
            ###src4 = src4.transpose(2, 1)  # batchsize * dimension * T
            ###src5 = src1 + src4  # batchsize * dimension * T

            ##s#rc5 = src5.transpose(2, 1)
            ###src6 = src5.transpose(1, 0)
            ###src6 = src6.transpose(2, 1)
            ###src7 = src6.transpose(2, 1)
            ###src8 = src7.transpose(2, 0)
            ##suncanrunsrc = src.transpose(2, 1)  #
        #src = src.transpose(1, 0)
        #src = src.transpose(2, 1)
         # dim=128
        ############################ self-attention
        #if self.is_AudioTransformer:
           #src = self.AudioTransformer(src)

        #src = self.AudioTransformer(src)
        #src = src.transpose(2, 1)
        #src = src.transpose(1, 0)    #dim batchsize time
        #src = src.transpose(2, 0)

        if not self.decoder_only:
            src = src * math.sqrt(self.nhid)
            src = self.pos_encoder(src)
            src = self.transformer_encoder(src, None)
        #src = {'src8': src8, 'src_vlad': src_vlad}
        src_new = {'src6': src, 'src_layer2': src_layer2, 'src': src}  # change at here using the high-dimensional or low-dimensional featrue
        return src_new

    def decode(self, mem, tgt, src, input_mask=None, target_mask=None, target_padding_mask=None):
        # tgt:(batch_size, T_out)
        # mem:(T_mem, batch_size, nhid)
        # sun: src_vlad(batch_size, times, dimension) need to transfer as (times, batch_size, dimension)

        tgt = tgt.transpose(0, 1)
        if target_mask is None or target_mask.size()[0] != len(tgt):
            device = tgt.device
            target_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)

        tgt = self.word_emb(tgt) * math.sqrt(self.nhid)

        tgt = self.pos_encoder(tgt)

        output1 = self.transformer_decoder(tgt, mem,
                                           memory_mask=input_mask,
                                           tgt_mask=target_mask,
                                           tgt_key_padding_mask=target_padding_mask)

        output2 = self.transformer_decoder(tgt, src,
                                           memory_mask=input_mask,
                                           tgt_mask=target_mask,
                                           tgt_key_padding_mask=target_padding_mask)

        # here adding vlad output: src_vlad   dim:(bathsize, times, dimension)
        #output3 = output1 + output2  # feature confusion   (output1)high + (output2)lower dimension feature

        #output = self.dec_fc(output3)

        output1 = self.dec_fc(output1)
        output2 = self.dec_fc(output2)
        output = {'output1': output1, 'output2': output2}
        return output

    def forward(self, src, tgt, src_vlad, input_mask=None, target_mask=None, target_padding_mask=None):

        src_new = self.encode(src)
        output = self.decode(src_new['src6'], tgt, src_new['src'],
                             input_mask=input_mask,
                             target_mask=target_mask,
                             target_padding_mask=target_padding_mask)
        return output





