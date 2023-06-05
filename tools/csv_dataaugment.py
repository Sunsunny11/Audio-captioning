import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import librosa
from re import sub
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import time
from itertools import chain
from torchvision.transforms import ToTensor

from tools.dataset import _sentence_process
from tools.file_io import load_csv_file, write_pickle_file, write_csv_file, flatten
import csv
import pickle
#from csv_dataaugment import _sentence_process
from tools.mainaugment import unmasker_instead, length, unmasker_insert, translation

#from transformers import pipeline
#import random



#train_loader = pd.read_csv('./data/csv_files/clotho_captions_development.csv')
#train_loader.columns = ['file_name', 'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']

#print(train_loader)     #print all the content
#print(train_loader[0:2])  #print the first two arrray

#l = load_metadata(Clotho, load_csv_file)


#if dataset == 'AudioCaps' and 'train' in csv_file:
    #caption_field = None
#else:
    #caption_field = ['caption_{}'.format(i) for i in range(1, 6)]

#unmasker = pipeline('fill-mask', model='bert-base-cased')

csv_list = load_csv_file('/vol/research/AAC_CVSSP_research/js0129/DCASE2022_task6_a-main/data/Clotho/csv_files/train.csv') #3839
#csv_list2 = load_csv_file('/vol/research/AAC_CVSSP_research/js0129/DCASE2022_task6_a-main/tools/trainAugment.csv')
audio_names = []
captions = []
augment = []
trainAugment = []

device, device_name = (torch.device('cuda'), torch.cuda.get_device_name(torch.cuda.current_device()))



for i, item in enumerate(csv_list):  #item shows the context of one wav file and the five captions
    caption_field = ['caption_{}'.format(i) for i in range(1, 6)]  #show the five captions
    audio_name = item['file_name']    #shows .wav file
    if caption_field is not None:
        item_captions = [_sentence_process(item[cap_ind], add_specials=False) for cap_ind in caption_field]
        for n in item_captions:
            n = n.cuda()
            #ntensor = ToTensor(n)
            item_captions_augment = unmasker_insert(n).cuda()
            #item_captions_augment = unmasker_instead(n)
            item_captions_augment = item_captions_augment.to(device)
            augment.append(item_captions_augment)
            augments = augment.cuda()
    else:   #sentence, add_specials=False
        item_captions = _sentence_process(item['caption'])
    audio_name = [audio_name]
    item_captions.extend(augments)
    item_captions = item_captions.to(device)
    audio_name.extend(item_captions)
    audio_name = audio_name.cuda()
    #trainAugment.append([audio_name, item_captions, augments])
    #result = []
    #ftrainAugment = list(flatten(trainAugment))
    #f = write_csv_file(trainAugment, column_name)
    #f = nested_list(trainAugment)
    augment = []
    audio_names.append(audio_name)
    audio_names = audio_names.cuda()
    #captions.append(item_captions) #shows five captions like 'a muddled noise of broken channel of the tv','a television blares','loud relevision static','heavy','heavy of the TV'

column_name = ['file_name', 'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5', 'caption_6', 'caption_7', 'caption_8', 'caption_9', 'caption_10']
csv_name = 'trainAugmentcuda.csv'
xml_df = pd.DataFrame(audio_names, columns=column_name)
xml_df.to_csv(csv_name, index=None)

#meta_dict = {'audio_name': np.array(audio_names), 'captions': np.array(captions)}
#np.savetxt('trainaugment.csv', meta_dict)

#def load_csv_file(file_name):
    #with open(file_name, 'r') as f:
        #csv_reader = csv.DictReader(f)
        #csv_obj = [csv_line for csv_line in csv_reader]
   # return csv_obj

#def write_csv_file(csv_obj, file_name):
    #with open(file_name, 'w') as f:
        #writer = csv.DictWriter(f, csv_obj[0].keys())
        #writer.writeheader()
        #writer.writerows(csv_obj)
    #print(f'Write to {file_name} successfully.')

