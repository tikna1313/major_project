'''
import numpy as np
import cv2
import os

for k in os.listdir('stage1_train'):
    img = np.zeros([256, 256, 3])
    for im in os.listdir('stage1_train/'+k+'/masks'):
        mask = cv2.imread('stage1_train/'+k+'/masks/'+im)
        mask = cv2.resize(mask, (256, 256))
        print(img.shape, mask.shape)
        img = np.maximum(img, mask)
    cv2.imwrite('combined/'+k+'.png', img)
'''

"""
import numpy as np
import cv2
import os

for k in os.listdir('stage1_train'):
    for im in os.listdir('stage1_train/'+k+'/images'):
        mask = cv2.imread('stage1_train/'+k+'/images/'+im)
        mask = cv2.resize(mask, (256, 256))
        print(mask.shape)
        cv2.imwrite('resized_images/' + k + '.png', mask)
"""

import pickle
import os
import cv2
path = os.listdir('resized_images')
images = []
for im in path:
    images.append(cv2.imread('resized_images/'+im))

print(len(images))

with open('mask_images.m', 'wb') as f:
    pickle.dump(images, f)