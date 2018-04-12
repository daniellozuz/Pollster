# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:37:14 2018

@author: OMEN
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import multiprocessing

def CaptureFrame ():
    V_list = []
    IsCaptured = 0
    
    #Choosing source of input
    
    #cap = cv2.VideoCapture(0)
    
    cap = cv2.VideoCapture("ankieta_OK.avi"); 
    
    while(True):
    #   skoro chcecie kazda ramke
    #    if  not(i % 5): 
        ret, frame = cap.read()
        if ret == 1 :
            
            #Converting image to HSV
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            V = np.mean(frame_HSV[:,:,2])
            #Aggregating V-values
            V_list.append(V)
            if IsCaptured == 0 and V>80 and i>10 and np.sum(np.square(V - V_list[-10:-2]))<1:
                #pipe.send(frame)
                V_list.clear
                IsCaptured = 1
            if IsCaptured == 1 and V < 10:
                IsCaptured = 0
        else:
            break
            
    #plt.plot(V_list)
    cap.release()
    cv2.destroyAllWindows()
    

