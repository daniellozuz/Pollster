#!/usr/bin/python3

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import json
import multiprocessing
import time
import os
from collections import deque
import math


def CaptureFrame (name,wait,thresh):
    V_list = []
    IsCaptured = 0
    
    #Choosing source of input
    
    #cap = cv2.VideoCapture(0)
    
    cap = 1
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
            if IsCaptured == 0 and V>thresh and i>10 and np.sum(np.square(V - V_list[-10:-2]))<1:
                while(czekaj<wait):
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
        if ((box[2, 0]-box[3,0])==0):
            kat = 0
        else:
            x =(box[3, 1] - box [2, 1])/(box[2,0]-box[3,0]) 
            kat = math.atan(x)
            kat = np.degrees(kat)
    else:
        if ((box[2, 0]-box[3,0])==0):
            kat = 0
        else:
            x =(box[0, 0] - box [1, 0])/(box[0,1]-box[1,1])
            kat = math.atan(x)
            kat = np.degrees(kat)
            
    rows = I.shape[0]
    cols = I.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-kat,1)
    I = cv2.warpAffine(I,M,(cols,rows))

    if(box[indeks,0] > 200):
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
    print('img_gray', img_gray)
    edges = cv2.Canny(img_gray,edge_min,edge_max)
    print(edges)
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
    cv2.equalizeHist(img, img)
    ret, img = cv2.threshold(img, 40, 150, 0)
    kernel = np.ones((4, 4), np.uint8)
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

def main(frame):
    #podawana jest sciezka do pliku z nagraniem sekwencji 
    I=frame
    #złapana ramka jest wywietlana
    # cv2.imshow("ankieta", np.uint8(I))
    # cv2.waitKey(10)
    
    #z ramki wycinana jest sama ankieta, niwelacja przesunięć kątowych 
    img,I_cnt = fixture (I, 70, 255)
    print(img)
    # cv2.imshow("Wycieta ankieta", np.uint8(img))
    # cv2.imshow("cnt", np.uint8(I_cnt))
    img_gray =  cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)

    #nalezy podać ciezkę do wzoru znacznika pozycji 
    loc = []
    template = cv2.imread('templates/pozycja.png')
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
    # cv2.imshow('wynik',np.uint8(img))
    maska = cv2.imread('templates/mask.png')
    maska =  cv2.cvtColor(np.uint8(maska), cv2.COLOR_BGR2GRAY)
        
    #Wycinam pola odpowiedzi z obrazu na podstawie maski 
    #Tu odbywa się ujednolicenie wielkoci i zwracane są trzy wycinki:
    #pole odpowiedzi, maska pola odpowiedzi, pole kodu, maska pola kodu
    [odpowiedzi, odpowiedzi_mask, kod, kod_mask ] = find_fields(img_gray, maska, centers)
    
    #wykrywam, które z odpowiedzi są zapisane i zwracam ich liste
    punkty = []
    punkty = ans_detection (odpowiedzi, odpowiedzi_mask)
    
    #wykrywam, które pola kodu są zapisane i zwracam ich liste 
    kody = []
    kody = kod_detection (kod, kod_mask)
    
    return punkty, kody 

#wywolanie funkcji głównej:
    
#Hej, jesli chciałby wywołać funkcję main, to musisz w jej kodzie zmienić
    #scieżki do maski i to wzoru znacznika, które powinny być załączone z programem 
#kody to jest: kolumna, numer, współrzędne, jasnosc, wynik
#punkty to jest: pytanie, odpowiedź, współrzędne, jasnosc, wynik

CAMERA_SETTINGS = {
    cv2.CAP_PROP_BRIGHTNESS: 151,
    cv2.CAP_PROP_CONTRAST: 80,
    cv2.CAP_PROP_SATURATION: 12,
    cv2.CAP_PROP_HUE: 13,
    cv2.CAP_PROP_EXPOSURE: -5,
}

class Application(tk.Frame):
    WAITING_FOR_EMPTY_CHAMBER = 0
    WAITING_FOR_NEXT_POLL = 1
    TEMPLATE_CREATION = 0
    POLLING = 1
    DEQUE_LENGTH = 40
    MAX_VARIANCE = 2
    LOWER_BRIGHTNESS = 5
    UPPER_BRIGHTNESS = 80

    def __init__(self, pipe_source, master=None):
        super().__init__(master)
        self.v_means = deque(maxlen=Application.DEQUE_LENGTH)
        self.pipe_source = pipe_source
        self.state = Application.WAITING_FOR_EMPTY_CHAMBER
        self.mode = Application.TEMPLATE_CREATION
        self.cap = cv2.VideoCapture('whole_video/anka.avi')
        # self._setup_camera(CAMERA_SETTINGS)
        self.boxes = []
        self.box = {}
        self.pack()
        self.create_widgets()
        self.video_loop()

    def _setup_camera(self, camera_settings):
        for option, value in camera_settings.items():
            self.cap.set(option, value)

    def create_widgets(self):
        self.make_snapshot_button = tk.Button(self)
        self.make_snapshot_button['text'] = 'Make snapshot'
        self.make_snapshot_button['command'] = self.make_snapshot
        self.make_snapshot_button.pack(side='top')

        self.remove_last_box_button = tk.Button(self)
        self.remove_last_box_button['text'] = 'Remove last box'
        self.remove_last_box_button['command'] = self.remove_last_box
        self.remove_last_box_button.pack(side='top')

        self.save_template_button = tk.Button(self)
        self.save_template_button['text'] = 'Save template'
        self.save_template_button['command'] = self.save_template
        self.save_template_button.pack(side='top')

        self.start_pollster_button = tk.Button(self)
        self.start_pollster_button['text'] = 'Start pollster'
        self.start_pollster_button['command'] = self.start_pollster
        self.start_pollster_button.pack(side='top')

        self.question_number = tk.Entry(self.master)
        self.question_number.pack(side='top')

        self.template_name = tk.Entry(self.master)
        self.template_name.pack(side='top')

        self.live_footage = tk.Label(self.master)
        self.live_footage.pack(padx=10, pady=10, side='left')
        self.live_footage.place(x=20, y=220, width=640, height=480)

        self.snapshot = tk.Canvas(self.master, cursor='cross')
        self.snapshot.pack(padx=10, pady=10, side='left')
        self.snapshot.place(x=720, y=220, width=640, height=480)
        self.snapshot.bind('<ButtonPress-1>', self.on_button_press)
        self.snapshot.bind('<B1-Motion>', self.on_move_press)
        self.snapshot.bind('<ButtonRelease-1>', self.on_button_release)

        self.quit_button = tk.Button(self, text='QUIT', fg='red', command=self.quit)
        self.quit_button.pack(side='bottom')

    def quit(self):
        self.pipe_source.send('STOP')
        self.master.destroy()
        print('Thank you for using Pollster.')

    def video_loop(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self.imgtk = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.imgtk = Image.fromarray(self.imgtk)
            self.imgtk = ImageTk.PhotoImage(image=self.imgtk)
            self.live_footage.imgtk = self.imgtk
            self.live_footage.config(image=self.imgtk)
            if self.mode == Application.POLLING:
                v_mean = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,2])
                print(v_mean)
                self.v_means.append(v_mean)
                if all([self.state == Application.WAITING_FOR_NEXT_POLL,
                        v_mean > Application.UPPER_BRIGHTNESS,
                        np.var(self.v_means) < Application.MAX_VARIANCE]):
                    print(os.getpid(), 'Sending poll to be measured for brightness.')
                    self.pipe_source.send(frame)
                    self.state = Application.WAITING_FOR_EMPTY_CHAMBER
                if all([self.state == Application.WAITING_FOR_EMPTY_CHAMBER,
                        v_mean < Application.LOWER_BRIGHTNESS]):
                    self.state = Application.WAITING_FOR_NEXT_POLL
            else:
                self.v_means.clear()
        else:
            self.pipe_source.send('STOP')
        self.master.after(20, self.video_loop)
    
    def start_pollster(self):
        if self.mode == Application.TEMPLATE_CREATION:
            self.mode = Application.POLLING
            self.start_pollster_button.config(background='green')
        elif self.mode == Application.POLLING:
            self.mode = Application.TEMPLATE_CREATION
            self.start_pollster_button.config(background='red')

    def make_snapshot(self):
        self.template_image = self.frame
        self.snapshot.copy_image = self.imgtk
        self.snapshot.create_image(0, 0, anchor='nw', image=self.snapshot.copy_image)

    def remove_last_box(self):
        if self.boxes:
            self.snapshot.delete(self.boxes[-1]['id'])
            self.boxes.pop()
            print('Last box removed.')
        
    def save_template(self):
        cv2.imwrite('templates/{}.png'.format(self.template_name.get()), self.template_image)
        with open('templates/{}.txt'.format(self.template_name.get()), 'w') as boxes_file:
            json.dump(self.boxes, boxes_file, indent=2)
        print('Template saved.')

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.box['question'] = self.question_number.get()
        self.box['id'] = self.snapshot.create_rectangle(event.x, event.y, event.x, event.y, outline='red')

    def on_move_press(self, event):
        cursor_x = event.x
        cursor_y = event.y
        self.snapshot.coords(self.box['id'], self.start_x, self.start_y, cursor_x, cursor_y)

    def on_button_release(self, event):
        self.box['coordinates'] = self.snapshot.coords(self.box['id'])
        self.boxes.append(self.box.copy())
        print('boxes:', self.boxes)
        for box in self.boxes:
            print(box['question'], self.snapshot.coords(box['id']))

class PollRecogniser(object):
    def receive_polls(self, pipe_target):
        while True:
            frame = pipe_target.recv()
            print(frame)
            cv2.imshow('dupa', frame)
            cv2.waitKey(2000)
            punkty, kody = main(frame)
            print(punkty, kody)
            if frame == 'STOP':
                print('Stopping poll recognition.')
                break
            print(os.getpid(), 'Frame came.')

def run_application(pipe_source):
    root = tk.Tk()
    root.geometry('1400x780+50+50')
    app = Application(pipe_source, master=root)
    app.mainloop()

def run_recogniser(pipe_target):
    consumer = PollRecogniser()
    consumer.receive_polls(pipe_target)


if __name__ == '__main__':
    pipe_target, pipe_source = multiprocessing.Pipe()

    app_process = multiprocessing.Process(target=run_application,
                                          args=(pipe_source, ))
    consumer_process = multiprocessing.Process(target=run_recogniser,
                                               args=(pipe_target, ))

    app_process.start()
    consumer_process.start()
    app_process.join()
    consumer_process.join()
