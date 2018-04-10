# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:58:22 2018

@author: Chastine
"""

category = 'kebakaran'

#with open('dataset/'+category+'_removed_duplicate.json', 'r') as rfp:
#  with open('dataset/'+category+'.json', 'w') as wfp:
#    for line in rfp:
#      if category in line:
#        wfp.write(line)

with open('dataset-cutted/'+category+'_preprocessed.json', 'r') as rfp:
  with open('dataset-cutted/'+category+'.json', 'w') as wfp:
    for line in rfp:
      if category in line:
        wfp.write(line)