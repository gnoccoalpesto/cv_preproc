#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import imutils
import time
import pprint
import random
import scipy.interpolate
import matplotlib.pyplot as plt

frame = cv2.imread('/home/lar/Downloads/test4.png',0)

mask = np.ones(frame.shape)
point_list = []

def sortFirst(val): 
    return val[1]

while(True):
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    white_pixels = np.nonzero(frame)
    white_pixels = np.transpose(np.array(white_pixels))

    if len(white_pixels) == 0:
        # pprint.pprint(np.array(point_list))
        print("NO MORE POINTS")
        point_list.sort(key=sortFirst)
        pprint.pprint(np.array(point_list))
        A = np.array(point_list)
        y = A[:,0]
        # x = np.random.rand(10,1)
        
        x = A[:,1]
        # y = np.random.rand(10,1)
        pprint.pprint(x)
        pprint.pprint(y)            
        
        tck = scipy.interpolate.splrep(x,y, k=2, s=300)
        x2 = np.linspace(x[0], x[-1], 200)
        y2 = scipy.interpolate.splev(x2, tck)
        plt.plot(x, y, 'o', x2, y2)
        plt.show()

    point = random.choice(white_pixels)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        print(point)
        point_list.append(point)
        cv2.circle(frame, (point[1],point[0]), 45, (0, 0, 0), -1)
        cv2.circle(mask, (point[1],point[0]), 45, (0, 225, 0), -1)

    if key == ord('q'):
        print(point_list)
        break

cv2.destroyAllWindows()