# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:37:14 2018

@author: Swoboda

Przerobilem troche CaptureFrame na potrzeby mojej pracy, plus dodalem czekanie po znalezieniu ok
ramki, bo byp palec na zdjeciach. To z palcem trzeba wyeliminowac, a reszte jakos wkleic do reszty :D

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
#import multiprocessing

def CaptureFrame (name):
    V_list = []
    IsCaptured = 0
    
    #Choosing source of input
    
    #cap = cv2.VideoCapture(0)
    
    cap = cv2.VideoCapture(name); 
    i = -1
    lista =[]
    while(True):
        czekaj = 0 

        i=i+1
    #   skoro chcecie kazda ramke
    #    if  not(i % 5): 
        ret, frame = cap.read()
        if ret == 1 :
            
            #Converting image to HSV
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            V = np.mean(frame_HSV[:,:,2])

            #Aggregating V-values
            V_list.append(V)
            if IsCaptured == 0 and V>90 and i>10 and np.sum(np.square(V - V_list[-10:-2]))<1:
                while(czekaj<10):
                    ret, frame = cap.read()
                    czekaj = czekaj +1
                lista.append(frame)
                V_list.clear
                IsCaptured = 1
            if IsCaptured == 1 and V < 10:
                IsCaptured = 0
        else:
            break
            
    #plt.plot(V_list)
    cap.release()
    cv2.destroyAllWindows()
    return lista

#Ta funkcja dostaje ramkę i progi 
def fixture (img, low, up):
    I=img
    imgray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,low,up,0)
    kernel = np.ones((9,9),np.uint8)
    img_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)
    im2, contours, hierarchy = cv2.findContours(img_close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    areat = 0
    for i in range (0,len(contours)): 
        area = cv2.contourArea(contours[i])
        if(area> areat):
            areat = area
            cnt = contours[i]
        
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(I,[box],0,(0,0,255),2)

    #Tu licze kat o jaki to jest przesuniete , ale dosyc chujowo , trzeba to sprawdzic
    x =( box[ 0, 0] - box [1, 0])/(box[0,1]-box[1,1])
    kat = math.atan(x)
    kat = kat* 57.29577951308
    
    rows = I.shape[0]
    cols = I.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-kat,1)
    I = cv2.warpAffine(I,M,(cols,rows))

    X= (box[ 0, 0] + box [1, 0])/2
    Y= (box[1,1]+box[2,1])/2
    X2= (box[2,0]+box[3,0])/2 
    Y2= (box[0,1]+box[3,1])/2

    newim = np.zeros([ np.int(Y2-Y), np.int(X2-X),3])

    for i in range(0, np.int(Y2-Y)):
        for j in range(0, np.int(X2-X)):
            if(i<10 or j<10 or j>X2-X-10 or i>Y2-Y-10):
                newim[i,j] = 255
            else:
                newim[i,j]=I[i+np.int(Y),j+np.int(X)]
        
    return newim

def preproc (img, low, up):
    imgray =  cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,low,up,0)
    cv2.equalizeHist( thresh, thresh );
    kernel = np.ones((4,4),np.uint8)
    kernel1 = np.ones((2,2),np.uint8)
    kernel2 = np.ones((1,1),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
    img_open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_OPEN, kernel)
    #img_close= cv2.erode(img_close,kernel1,iterations = 1)
     
    return img_close

def myabsdiff (imgf, imgf2):
        
    if(imgf.shape[0]<imgf2.shape[0]):
        row=imgf.shape[0]
    else:
        row=imgf2.shape[0]
    if(imgf.shape[1]<imgf2.shape[1]):
        col=imgf.shape[1]
    else:
        col=imgf2.shape[1]

    diff = np.zeros([ row, col])

    for i in range(0, row):
        for j in range(0, col):
            diff[i,j] = np.abs(imgf[i,j] - imgf2[i,j])
    
    
    return diff
    

I=CaptureFrame("C:\\Users\kSwoboda\\Desktop\\ankieter\\Pollster-master\\ICK_ankieter\\ankieta_OK.avi")
wzor =CaptureFrame("C:\\Users\kSwoboda\\Desktop\\ankieter\\Pollster-master\\ICK_ankieter\\wzor.avi")
I=I[0]
wzor = wzor[0]

#cv2.imshow("wzor", wzor)
#cv2.imshow("frame", I)

img = fixture (I, 20, 255)
cv2.imshow("cutted", np.uint8(img))
imgf = preproc (img, 100, 255)
cv2.imshow("filtered", np.uint8(imgf))

img2= fixture (wzor, 20, 255)
cv2.imshow("cutted2", np.uint8(img2))
imgf2 = preproc (img2, 100, 255)
cv2.imshow("filtered2", np.uint8(imgf2))

#testy tych obrazow diff
diff = myabsdiff (imgf, imgf2)
cv2.imshow("diff", np.uint8(diff))

img =  cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
img2 =  cv2.cvtColor(np.uint8(img2), cv2.COLOR_BGR2GRAY)
ret,img = cv2.threshold(img,100,255,0)
ret,img2 = cv2.threshold(img2,100,255,0)

diff = myabsdiff (img,img2)
cv2.imshow("diff2", np.uint8(diff))

cv2.waitKey(10)
    

