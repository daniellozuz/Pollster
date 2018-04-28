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
        self.cap = cv2.VideoCapture('whole_video/all.avi')
        self.boxes = []
        self.box = {}
        self.pack()
        self.create_widgets()
        self.video_loop()

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
