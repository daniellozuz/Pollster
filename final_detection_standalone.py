# -*- coding: utf-8 -*-
"""
Created on Tue May 22 21:52:31 2018

@author: kSwoboda
"""

import numpy as np
import cv2
import math
import os
import pandas as pd
from operator import itemgetter

def CaptureFrame (name,wait,thresh):
    V_list = []
    IsCaptured = 0
    
    #Choosing source of input
    
    cap = cv2.VideoCapture(0)
    
   # cap = cv2.VideoCapture(name); 
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

            print(V)
            #Aggregating V-values
            V_list.append(V)
            if IsCaptured == 0 and V>thresh and i>10 and np.sum(np.square(V - V_list[-10:-2]))<1:
                while(czekaj<wait):
                    ret, frame = cap.read()
                    czekaj = czekaj +1
                lista.append(frame)
                
                V_list.clear
                break
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
    cv2.imshow("binaryzacja", np.uint8(img_close))
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
    

    #Tu licze kat o jaki to jest przesuniete 
    minx = 10000 
    indeks = -1
    for i in range (0,4): 
        if(box[i,1]<minx):
            minx = box[i,1]
            indeks = i
            
    if(box[indeks,0] <200):
        if((box[2,0]-box[3,0])==0):
            kat = 0
            
        else:
            x =(box[3, 1] - box [2, 1])/(box[2,0]-box[3,0]) 
            kat = math.atan(x)
            kat = np.degrees(kat)
            
    else:
        if((box[2,0]-box[3,0])==0):
            kat = 0
            
        else:
            x =(box[0, 0] - box [1, 0])/(box[0,1]-box[1,1])
            kat = math.atan(x)
            kat = np.degrees(kat)
    
    
    rows = I.shape[0]
    cols = I.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-kat,1)
    I = cv2.warpAffine(I,M,(cols,rows))

    if(box[indeks,0]> 200 ):
        X= (box[0, 0] + box [1, 0])/2
        Y= (box[1,1]+box[2,1])/2
        X2= (box[2,0]+box[3,0])/2 
        Y2= (box[0,1]+box[3,1])/2
    else:
        X= (box[2, 0] + box [1, 0])/2
        Y= (box[3,1]+box[2,1])/2
        X2= (box[0,0]+box[3,0])/2 
        Y2= (box[0,1]+box[1,1])/2

    newim = np.zeros([ np.int(Y2-Y), np.int(X2-X),3])

    for i in range(0, np.int(Y2-Y)):
        for j in range(0, np.int(X2-X)):
            if(i<5 or j<5 or j>X2-X-5 or i>Y2-Y-5):
                newim[i,j] = 255
            else:
                newim[i,j]=I[i+np.int(Y),j+np.int(X)]
        
    return newim,I

def find_templates (img_gray, edge_min, edge_max, template_treshold, template): # tu dodatkowo wejdzie : template
# przykładowe parametry: edge_min = 100 , edge_max = 200, template_treshold = 0,5
# funkcja zwraca unikatową listę wykryć wzoru 

    edges = cv2.Canny(img_gray,edge_min,edge_max)
    img_gray = np.uint8(edges)

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
    
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

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
            warunek = (((cnt_sorted[c][0][0][0]>= loc[i][1]) & (cnt_sorted[c][0][0][0]<= (loc[i][1] +template_width)) & (cnt_sorted[c][0][0][1]>= loc[i][0]) & (cnt_sorted[c][0][0][1]<= (loc[i][0] +template_height))))
            if(warunek):
                cnt_ok.append(cnt_sorted[c])
                break
#usunę duplikaaty konturów bo to dużo psuje, ale nie wiem czemu są i czemu psują 
    cnt_ok2 = []           
    cnt_sorted2 = sorted(cnt_ok, key=lambda x: x[0][0][1])
    cnt_ok2.append(cnt_sorted2[0])
    for c in range (1,len(cnt_sorted2)): 
        if(not((cnt_sorted2[c][0][0][0] == cnt_sorted2[c-1][0][0][0]) & (cnt_sorted2[c][0][0][1] == cnt_sorted2[c-1][0][0][1]))):
            cnt_ok2.append(cnt_sorted2[c])
    return cnt_ok2

def wycinek (img_gray, edge_min, edge_max,):
    edges = cv2.Canny(img_gray,edge_min,edge_max)
    img_gray = np.uint8(edges)

    template = np.zeros([ 15, 15])
    for i in range(256, 271):
        for j in range(352, 367):
            template[i-256,j-352] = edges[i][j]
    template = np.uint8(template)
    return template

def marker_center(cnt):
    cnt_sorted = sorted(cnt, key=lambda x: x[0][0][0])
    # wyznaczam i sortuje srodki znaczników 
    centers = []
    for i in range (0,len(cnt)):
        M = cv2.moments(cnt_sorted[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if(i%2 == 0):
            temp = [cx , cy]
        else:   
            if(cy>temp[1]):
                centers.append(temp)
                centers.append([cx,cy])
            else:
                centers.append([cx,cy])
                centers.append(temp)
    
    return centers
    
def find_fields(img, maska, centers):
    
    #preprocessing obrazu
    cv2.equalizeHist( img, img );
    ret,img = cv2.threshold(img,40,150,0)
    kernel = np.ones((4,4),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    #Wycięcie obszarów do detekcji:
    
    #obraz: wycinek z pozycji
    X = centers[0][0]
    Y = centers[0][1] 
    W = centers[2][0] - centers[0][0]
    H = centers[1][1] - centers[0][1]
    odpowiedzi = img[Y:Y+H,X:X+W]

        
    #maska: wycinek z pozycji 
    X = 29
    Y = 25
    W = 307
    H = 293
    odpowiedzi_mask = maska[Y:Y+H,X:X+W]
    #obraz: wycinek z pozycji: kod
    X = centers[4][0]
    Y = centers[4][1] 
    W = centers[6][0] - centers[4][0]
    H = centers[5][1] - centers[4][1]
    kod = img[Y:Y+H,X:X+W]
    #maska: wycinek z pozycji: kod
    X = 356
    Y = 168
    W = 121
    H = 86
    kod_mask = maska[Y:Y+H,X:X+W]
    
    odpowiedzi = cv2.resize(odpowiedzi, (odpowiedzi_mask.shape[1],odpowiedzi_mask.shape[0]))
    kod = cv2.resize(kod, (kod_mask.shape[1],kod_mask.shape[0]))    
    
    return odpowiedzi, odpowiedzi_mask, kod, kod_mask 
    
def ans_detection (odpowiedzi, odpowiedzi_mask):
    #tworzę obraz złączony maski i wycinku 
    odpowiedzi_det = np.ones([ odpowiedzi.shape[0], odpowiedzi.shape[1]]) 
    odpowiedzi_det = odpowiedzi_det*255
    for i in range (0, odpowiedzi.shape[0]):
        for c in range (0,odpowiedzi.shape[1]):
            if(odpowiedzi_mask[i][c] >200):
                odpowiedzi_det[i][c]= odpowiedzi[i][c]
    odpowiedzi_det = 255-odpowiedzi_det 
    odpowiedzi_show =  cv2.cvtColor(np.uint8(odpowiedzi_det), cv2.COLOR_GRAY2RGB)
        
    im2, contours, hierarchy = cv2.findContours(odpowiedzi_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt_sorted = sorted(contours, key=lambda x: (x[0][0][1],-x[0][0][0]))
    punkty = []
    
    cv2.drawContours( odpowiedzi_show,cnt_sorted,0,(0,0,255),2)
    
    #punkty to jest: pytanie, odpowiedź, współrzędne, jasnosc, wynik
    pytania = 0 
    odpowiedzii = 0
    pytanie =1
    odpowiedz = 1
    addy = 0
    pytrow = 17
    for i in range (0,len(cnt_sorted)):
        M = cv2.moments(cnt_sorted[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        jasnosc = (np.sum(odpowiedzi_det[cy-3:cy+3,cx-3:cx+3]))/36
        if(jasnosc > 200):
            wynik = 1
            cv2.circle(odpowiedzi_show,(cx,cy), 2, (0,255,0), -1)
        else:
            wynik = 0
            cv2.circle(odpowiedzi_show,(cx,cy), 2, (0,0,255), -1)
        pytania = pytania + 1
        if(pytania== 186):
            pytrow = 16
            pytania = 16
        punkty.append([pytanie+addy,odpowiedz, cx, cy, jasnosc, wynik])
        if(pytania%pytrow == 0):
            odpowiedz = odpowiedz + 1
            odpowiedzii = odpowiedzii + 1
            pytanie = 0
        pytanie = pytanie + 1 
        if(odpowiedzii == 5):
            odpowiedzii = 0;
            odpowiedz = 1
            addy = addy +17
        cv2.imshow('odpowiedzi',np.uint8(odpowiedzi_show)) 
        cv2.waitKey(10)
    return punkty 

def kod_detection (kod, kod_mask):
    
    kod_det = np.ones([ kod.shape[0], kod.shape[1]]) 
    kod_det = kod_det*255
    for i in range (0, kod.shape[0]):
        for c in range (0,kod.shape[1]):
            if(kod_mask[i][c] >200):
                kod_det[i][c]= kod[i][c]
    kod_det = 255-kod_det 
    kod_show =  cv2.cvtColor(np.uint8(kod_det), cv2.COLOR_GRAY2RGB)
        
    im2, contours, hierarchy = cv2.findContours(kod_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt_sorted = sorted(contours, key=lambda x: (x[0][0][1],-x[0][0][0]))
    
    cv2.drawContours( kod_show,cnt_sorted,0,(0,0,255),2)
    
    kody = []
    #kody to jest: kolumna, numer, współrzędne, jasnosc, wynik
    pytania = 0 
    odpowiedzii = 0
    kolumna =1
    numer = 1
    pytrow = 9
    once = 1
    once2 = 1
    once3 = 1
    for i in range (0,len(cnt_sorted)):
        M = cv2.moments(cnt_sorted[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        jasnosc = (np.sum(kod_det[cy-4:cy+4,cx-2:cx+2]))/32
        if(jasnosc > 200):
            wynik = 1
            cv2.circle(kod_show,(cx,cy), 2, (0,255,0), -1)
        else:
            wynik = 0
            cv2.circle(kod_show,(cx,cy), 2, (0,0,255), -1)
        pytania = pytania + 1

        kody.append([kolumna,numer, cx, cy, jasnosc, wynik])
        if(pytania%pytrow == 0):
            numer = 0
            odpowiedzii = odpowiedzii + 1
            kolumna = kolumna + 1
        numer = numer + 1 
        if((odpowiedzii == 2) & once):
            once = 0
            pytania = 0
            pytrow = 10
            numer = numer -1
        if((odpowiedzii == 3)& once3):
            numer = numer -1
            once3 = 0
        if((odpowiedzii == 4) & once2):
            once2 = 0
            pytania = 0
            pytrow = 5
            numer = numer -1 
            
    cv2.imshow('kody',np.uint8(kod_show)) 
    cv2.waitKey(10)
    
    
    return kody

def main(I):
    #podawana jest sciezka do pliku z nagraniem sekwencji 

    #złapana ramka jest wywietlana
    cv2.imshow("ankieta", np.uint8(I))
    cv2.waitKey(10)
    
    #z ramki wycinana jest sama ankieta, niwelacja przesunięć kątowych 
    img,I_cnt = fixture (I, 70, 255)
    cv2.imshow("Wycieta ankieta", np.uint8(img))
    cv2.imshow("cnt", np.uint8(I_cnt))
    img_gray =  cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)

    #nalezy podać ciezkę do wzoru znacznika pozycji 
    loc = []
    template = cv2.imread('C:\\Users\kSwoboda\\Desktop\\pozycja.png')
    template =  cv2.cvtColor(np.uint8(template), cv2.COLOR_BGR2GRAY)
    w, h = template.shape
    #wzory są znajdowane, a następnie precyzyjnie okrelane są ich srodki (centers)
    loc = find_templates (img_gray, 150, 200, 0.57, template)
    cnt = []
    cnt = find_contours_in_templates (loc,img, w,h, 20, 255, 0, 200)
    centers = []
    centers = marker_center(cnt)
    
    #Następnie wczytywana jest maska pól z odpowiedziami 
    cv2.drawContours(img, cnt, -1, (255,255,255), 1)
    cv2.imshow('wynik',np.uint8(img))
    maska = cv2.imread('C:\\Users\kSwoboda\\Desktop\\mask.png')
    maska =  cv2.cvtColor(np.uint8(maska), cv2.COLOR_BGR2GRAY)
        
    #Wycinam pola odpowiedzi z obrazu na podstawie maski 
    #Tu odbywa się ujednolicenie wielkoci i zwracane są trzy wycinki:
    #pole odpowiedzi, maska pola odpowiedzi, pole kodu, maska pola kodu
    [odpowiedzi, odpowiedzi_mask, kod, kod_mask ] = find_fields(img_gray, maska, centers)
    
    #wykrywam, które z odpowiedzi są zapisane i zwracam ich liste
    punkty = []
    cv2.imshow('odp',np.uint8(odpowiedzi))
    punkty = ans_detection (odpowiedzi, odpowiedzi_mask)
    
    #wykrywam, które pola kodu są zapisane i zwracam ich liste 
    kody = []
    cv2.imshow('kodziki',np.uint8(kod))
    kody = kod_detection (kod, kod_mask)
    
    return punkty, kody 

#wywolanie funkcji głównej:
    
#Hej, jesli chciałby wywołać funkcję main, to musisz w jej kodzie zmienić
    #scieżki do maski i to wzoru znacznika, które powinny być załączone z programem 

#kody to jest: kolumna, numer, współrzędne, jasnosc, wynik
#punkty to jest: pytanie, odpowiedź, współrzędne, jasnosc, wynik

def has_duplicates(listObj):
    return len(listObj) != len(set(listObj))

def CreateColumnNames(file):
    #creating columns, in csv seperated by comas
    column_names = "Code, " + "".join(["Question " + str(i) + ", " for i in range(1,51)]) + "\n"
    file.write(column_names)
    
def GetCode(code):
    
    #getting list of only marked boxes
    marked = list(filter(lambda x: x[:][5] == 1, code))
    


    #getting the code
    name = ""
    
    for i in range(0,len(marked)):
        #checing if the code field was filled correctly
        if marked[i][0] != i:
            name += str(marked[i][1]) 
        else:
            return "Wrong Code!,"
    return name + ","
   
    
def GetAnswers(points):
    
    answers =  [[] for i in range(1,51)]
    #getting list of only marked boxes
    marked = list(filter(lambda x: x[:][5] == 1, points))
    #sorting by the questions
    marked = sorted(marked, key = itemgetter(0))
    
    #name is string containing answers
    #flag is used for preventing from saving error in multiple cells for one question
    idx = 0
    #getting the asnwers
    for i in range(0,len(marked)):
        idx = marked[i][0]
        answers[idx - 1].append(marked[i][1])
    return answers
        
  
#    
def ValidateAnswers(answers):
    
    validated = [[] for i in range(1,51)]
    for i in range(0,len(answers)):
        if len(answers[i]) > 1:
            validated[i] = 'Error'
        elif len(answers[i]) ==  0:
            validated[i] = "Empty, "
        else:
#                 converting index of filled box to letter indicating answer in form
            if answers[i][0] == 1:
                validated[i] = "A,"
            elif answers[i][0] == 2:
                validated[i] = "B,"
            elif answers[i][0] == 3:
                validated[i] = "C,"
            elif answers[i][0] == 4:
                validated[i] = "D,"
            elif answers[i][0] == 5:
                validated[i] = "E, "
                 
    #getting whole list into single string 
    return ''.join(str(r) for v in validated for r in v).replace(" ", "")
    
#main function saving data, the only one you need to use from this file
def SaveData(code, points):

    with open("out.csv", 'a') as csv:  
        #if file is empty initialize columns
        if os.path.getsize("out.csv") < 100:
            CreateColumnNames(csv)
        #read code from filled boxes
        form_data = GetCode(code)
        answers = GetAnswers(points)
        form_data += ValidateAnswers(answers)[0:-1] + "\n"
        csv.write(form_data)

        
    csv.close()
#To test SaveData function run final_detection.py, then uncommnet line below
while(1):
    nazwa = "C:\\Users\kSwoboda\\Desktop\\ankieter\\Pollster-master\\anka.avi"
    I=CaptureFrame(nazwa, 30, 80)
    I=I[0]
    [punkty, kody] = main(I)
    SaveData(kody,punkty)
