# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:26:11 2018

@author: Chastine
"""

import os
source1 = "train_augment"
dest11 = "test_augment"
files = os.listdir(source1)
import shutil
import numpy as np
for f in files:
    if np.random.rand(1) < 0.2:
        shutil.move(source1 + '/'+ f, dest11 + '/'+ f)