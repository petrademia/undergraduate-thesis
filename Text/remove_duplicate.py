# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:35:09 2018

@author: Petrus
"""

category = 'banjir'

file_toscan = 'dataset-cutted/'+category+'_preprocessed.json'

file_towrite = 'dataset-cutted/'+category+'_removed_duplicate.json'

def remove_dup_line():
    s = set()
    with open(file_towrite, 'w') as out:
        for line in open(file_toscan):
            if line not in s:
                out.write(line)
                s.add(line)
                
remove_dup_line()