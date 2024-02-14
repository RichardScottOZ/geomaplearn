# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:33:06 2023

@author: oakley
"""

import numpy as np
import matplotlib.pyplot as plt

history = np.load('DetectAreasAndAxisModel_history.npy',allow_pickle=True).item()

epochs = np.arange(len(history['loss']))
plt.plot(epochs,history['loss'],epochs,history['val_loss'])
plt.legend(['Training Loss','Validation Loss'])
plt.ylim(0.1,0.6)
plt.xlabel('Epoch')
plt.ylabel('Loss')