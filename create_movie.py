import os
import cv2


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('whole_video/all.avi',
                      fourcc,
                      15, (640, 480))

for file_name in sorted(os.listdir('src_video')):
    if file_name.startswith('OK'):
        print(file_name)
        cap = cv2.VideoCapture('src_video/' + file_name)
        
        while True:
            ret, frame = cap.read()
            #print(cap.isOpened())
            #print(frame.shape)
            if frame is None:
                break
            out.write(frame)

cap.release()
out.release()
