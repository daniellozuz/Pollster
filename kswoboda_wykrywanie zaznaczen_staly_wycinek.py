# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:34:10 2018

@author: kSwoboda
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
                while(czekaj<30):
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

def wycinek (img_gray, edge_min, edge_max,):
    edges = cv2.Canny(img_gray,edge_min,edge_max)
    img_gray = np.uint8(edges)

    template = np.zeros([ 14, 14])
    for i in range(287, 301):
        for j in range(124, 138):
            template[i-287,j-124] = edges[i][j]
    template = np.uint8(template)
    return template

# tu dodatkowo wejdzie : template i to trzeba dopisać , ale na razie sam się liczy 
def find_templates (img_gray, edge_min, edge_max, template_treshold, template): # tu dodatkowo wejdzie : template
# przykładowe parametry: edge_min = 100 , edge_max = 200, template_treshold = 0,5
# funkcja zwraca unikatową listę wykryć wzoru 

    edges = cv2.Canny(img_gray,edge_min,edge_max)
    img_gray = np.uint8(edges)

# tu jest liczony wycinek dodetekcji, nie będzie to robione tu, tylko ze wzoru( TO DO)
#    template = np.zeros([ 14, 14])
#    for i in range(287, 301):
#        for j in range(124, 138):
#            template[i-287,j-124] = edges[i][j]
#    template = np.uint8(template)
#koniec tej częsci, którą trzeba kiedy zmienić
    w, h = template.shape[::-1]
   
#Znajduje template w obrazie:
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

# Od tego momentu nastąpi kilka dziwnych operacji, ponieważ te template się czasem dublują 
    loc_repeat = np.where( res >= template_treshold)
    loc_trans = []
    for i in range (0, len(loc_repeat[1])):
        loc_trans.append((loc_repeat[0][i],loc_repeat[1][i],(loc_repeat[0][i]+loc_repeat[1][i])))

    from operator import itemgetter
    loc_repeat=sorted(loc_trans,key=itemgetter(2))

    loc = []
    loc.append([loc_repeat[0][0],loc_repeat[0][1]])
    for i in range (1, len(loc_repeat)):
        dist =  np.abs(loc_repeat[i][1] - loc_repeat[i-1][1]) + np.abs(loc_repeat[i][0] - loc_repeat[i-1][0])
        if(dist>10):
            loc.append([loc_repeat[i][0],loc_repeat[i][1]])
#loc zawiera już unikatowe template
    return loc

def find_contours_in_templates (loc,img, template_width, template_height, tresh_lowerB, tresh_upperB, cont_min, cont_max):
# przykładowe parametry: template_ width = 14, template_height = 14, tresh_upperB = 255, tresh_lowerB = 30, cont_min = 40, cont_max = 80
#poza tym , że funkcja przyjmuje w pizdu parametrów, to zwraca liste konturów, które nas obchodzą
    
    
#preprocessing obrazu
    img =  cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist( img, img );
    ret,img = cv2.threshold(img,tresh_lowerB,tresh_upperB,0)
    kernel = np.ones((1,1),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#wybranie odpowiedniej wielkosci konturow
    cnt = []
    for i in range (0,len(contours)): 
        area = cv2.contourArea(contours[i])
        if((area> cont_min) & (area<cont_max)):
            cnt.append(contours[i])

    cnt_ok = []
    cnt_sorted = sorted(cnt, key=lambda x: x[0][0][1])

#wybranie tych konturów, które leżą w obszarach template
    for i in range (0, len(loc)):
        for c in range (0,len(cnt)):
            war = (((cnt_sorted[c][0][0][0]>= loc[i][1]) & (cnt_sorted[c][0][0][0]<= (loc[i][1] +template_width)) & (cnt_sorted[c][0][0][1]>= loc[i][0]) & (cnt_sorted[c][0][0][1]<= (loc[i][0] +template_height))))
            if(war):
                cnt_ok.append(cnt_sorted[c])
                break
            
    return cnt_ok

def find_marks (img_gray, loc, cnt_ok,template_width, template_height, brightness_tresh):
#przykładowe parametry: template_width = 14, template_height =14, brightness_tresh =120
#funkcja zwraca listę punktów z wynikiem i jasnoscią
    
#tworzę maskę obrazu z zaznaczonymi polami
    mask = np.zeros([ img_gray.shape[0], img_gray.shape[1]])   
    mask = np.uint8(mask)
    mask= cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)     
    cv2.drawContours(mask, cnt_ok, -1, (255,255,255), -1)
    mask= cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 

    wypelnienie = []
    for i in range (0, len(loc)):
        jasnosc = 0
        out = 0 
        n = 0
        for j in range (loc[i][0],loc[i][0]+ template_width):
            for k in range (loc[i][1],loc[i][1]+ template_height):
                if(mask[j,k] > 1):
                    jasnosc = jasnosc + img_gray[j,k]
                    n= n+1
        if(n!=0):
            jasnosc = jasnosc /n
        if ( jasnosc < brightness_tresh):
            out = 1
        wypelnienie.append([out, jasnosc,[loc[i][0],loc[i][1]]])
    
    return wypelnienie


#ta funkcja wywołuje reszte, okrela parametry i robi wizualizacje    
def funkcja_test(nazwa):
    I=CaptureFrame(nazwa)
    I=I[0]
    img = fixture (I, 20, 255)
    cv2.imshow("Wycieta ankieta", np.uint8(img))
    img_gray =  cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
    
    Iw=CaptureFrame("C:\\Users\kSwoboda\\Desktop\\ankieter\\Pollster-master\\ICK_ankieter\\ankieta_OK.avi")
    Iw=Iw[0]
    imgw = fixture (Iw, 20, 255)
    imgw_gray =  cv2.cvtColor(np.uint8(imgw), cv2.COLOR_BGR2GRAY)
    
    template = wycinek (imgw_gray, 100, 200)
    
    #wielkosc wycinka
    w = 14
    h = 14
    
    loc = []
    loc = find_templates (img_gray, 100, 200, 0.43, template)
    cnt = []
    cnt = find_contours_in_templates (loc,img, w,h, 30, 255, 40, 80)
    wynik = []
    wynik = find_marks (img_gray, loc, cnt, w,h, 120)
    for i in range (0, len(loc)):
        pt = (loc[i][1], loc[i][0])
    
        if(wynik[i][0]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)
        else:
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
        
    cv2.drawContours(img, cnt, -1, (255,255,255), 1)
    cv2.imshow('wynik',np.uint8(img))
    cv2.waitKey(10)
    

#wywolanie funkcji testującej:
funkcja_test("C:\\Users\kSwoboda\\Desktop\\ankieter\\Pollster-master\\ICK_ankieter\\ankieta_Mix.avi")

    
    