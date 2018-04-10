# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:26:07 2018

@author: Petrus
"""

category = 'banjir'

raw_dataset = open('dataset/'+category+'_removed_duplicate.json', 'r')

processed_dataset = open('dataset/'+category+'_preprocessed.json', 'r')

raw_lines = raw_dataset.readlines()

raw_lines.sort()