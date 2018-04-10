# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:37:14 2018

@author: OMEN
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


i = 0
frame_list = []
V_list = []


#Choosing source of input
#cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture("ank4.mp4"); 

while(True):
    i += 1
    if  not(i % 5):
      
        #Capturing every 10th frame
        ret, frame = cap.read()
        if ret == 1 :
            cv2.imshow('frame',frame)
            # Aggregating frame
            frame_list.append(frame)
            
            #Converting image to HSV
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow('HSV',frame_HSV)
            #Aggregating V-values
            V_list.append(np.mean(frame_HSV[:,:,2]))

            #        if V_list[-1] < lower_limit
#            break
        else:
            break
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

V = []
for j in range(2,(len(V_list) - 3)):
    V.append(V_list[j] - 5*np.sum (np.square (V_list[j] - V_list[j-2:j+2])))


index_max_V = V.index(max(V))
I = frame_list[index_max_V+2]

cv2.imshow("MAX V VALUE",I)
plt.plot(V_list)

plt.show()



#cap2 = cv2.VideoCapture("ank4.mp4"); 
#
#while(True):
#    i += 1
#    if  not(i % 5):
#        if i == 52:
#        #Capturing every 10th frame
#            ret, frame = cap2.read()
#        if ret == 1 :
#            cv2.imshow('frame',frame)
#
#        break