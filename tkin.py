#!/usr/bin/python3

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import json

class Application(tk.Frame):
    WAITING_FOR_EMPTY_CHAMBER = 0
    WAITING_FOR_NEXT_POLL = 1
    TEMPLATE_CREATION = 0
    POLLING = 1

    def __init__(self, master=None):
        super().__init__(master)
        self.state = Application.WAITING_FOR_EMPTY_CHAMBER
        self.mode = Application.TEMPLATE_CREATION
        self.cap = cv2.VideoCapture('src_video/OK_a_wzor_sekwencja.avi')
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

        self.question_number = tk.Entry(root)
        self.question_number.pack(side='top')

        self.template_name = tk.Entry(root)
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

        self.quit = tk.Button(self, text='QUIT', fg='red', command=root.destroy)
        self.quit.pack(side='bottom')

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
                print('Sending poll to be measured for brightness.')
                # Place for sending it to another process responsible for calculating brightness (send it to poll consumer :-) )
        self.master.after(30, self.video_loop)
    
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

root = tk.Tk()
root.geometry('1400x780+50+50')
app = Application(master=root)
app.mainloop()
