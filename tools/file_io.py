#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import csv
import pickle
from collections import Iterable

def write_csv_file(csv_obj, file_name):

    with open(file_name, 'w') as f:
        writer = csv.DictWriter(f, csv_obj[0].keys())
        writer.writeheader()
        writer.writerows(csv_obj)
    print(f'Write to {file_name} successfully.')


def load_csv_file(file_name):

    with open(file_name, 'r') as f:
        csv_reader = csv.DictReader(f)
        csv_obj = [csv_line for csv_line in csv_reader]
    return csv_obj


def load_pickle_file(file_name):

    with open(file_name, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj


def write_pickle_file(obj, file_name):

    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Write to {file_name} successfully.')

#def nested_list(list_raw):
 #   for item in range(len(list_raw)):
  #      #if isinstance(item, list):
   #      items = item
    #    #else:
     #    items.append(items)
    #return items
def flatten(items, ignore_type=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_type):
            yield from flatten(x)
        else:
            yield x