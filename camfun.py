# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:45:10 2018

@author:Konrad Swoboda
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
def getviev():
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def saveframe(name):
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(name + '.avi',fourcc, 20.0, (640,480))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                # write the flipped frame
                out.write(frame)
                
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break  

def setopt():
        print('brightnes: %d' % cap.get(10))
        br= input('new:')
        cap.set(10,int(br))
        
        print('contrast: %d' % cap.get(11))
        br= input('new:')
        cap.set(11,int(br))
       
        print('saturation: %d' % cap.get(12))
        br= input('new:')
        cap.set(12,int(br))
        
        print('hue: %d' % cap.get(13))
        br= input('new:')
        cap.set(13,int(br))
        '''
        print('gain: %d' % cap.get(14))
        br= input('new:')
        cap.set(14,int(br))
        '''
        print('exposure: %d' % cap.get(15))
        br= input('new:')
        cap.set(15,int(br))
    
print ('q konczy program')        
userin= input('Chcesz zapisac(z), czy ogladac(o), jesli opcje to (zo lub oo): ')
if(userin == 'zo' or userin == 'oo'):
    setopt()
if (userin == 'o'or userin == 'oo'):
    getviev()
else: 
    if (userin == 'z' or userin == 'zo'):
        name = input('Daj nazwe pliku: ')
        saveframe(name)
    else:
        print('Nie umiesz czytac i pisac')
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()