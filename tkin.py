#!/usr/bin/python3

import tkinter as tk
import cv2
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.cap = cv2.VideoCapture('src_video/OK_a_wzor_sekwencja.avi')
        self.rectangles = []
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.snapshot_button = tk.Button(self)
        self.snapshot_button["text"] = "Make snapshot"
        self.snapshot_button["command"] = self.make_snapshot
        self.snapshot_button.pack(side="top")

        self.template_path = tk.Entry(root)
        self.template_path.pack(side="top")

        self.live_footage = tk.Label(self.master)
        self.live_footage.pack(padx=10, pady=10, side='left')
        self.live_footage.place(x=20, y=120, width=640, height=480)

        self.snapshot = tk.Canvas(self.master, cursor='cross')
        self.snapshot.pack(padx=10, pady=10, side='left')
        self.snapshot.place(x=720, y=120, width=640, height=480)
        self.snapshot.bind('<ButtonPress-1>', self.on_button_press)
        self.snapshot.bind('<B1-Motion>', self.on_move_press)
        self.snapshot.bind('<ButtonRelease-1>', self.on_button_release)

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.pack(side="bottom")
        self.video_loop()

    def video_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            self.frame = frame
            self.live_footage.imgtk = frame
            self.live_footage.config(image=frame)
        self.master.after(30, self.video_loop)

    def make_snapshot(self):
        print("Made snapshot.")
        self.snapshot.copy_image = self.frame
        self.snapshot.create_image(0, 0, anchor='nw', image=self.snapshot.copy_image)

    def on_button_press(self, event):
        print('Drawing at:', event.x, event.y)
        self.start_x = event.x
        self.start_y = event.y
        self.rectangles.append(self.snapshot.create_rectangle(event.x, event.y, event.x, event.y, outline='red'))

    def on_move_press(self, event):
        curX = event.x
        curY = event.y
        self.snapshot.coords(self.rectangles[-1], self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        print('Rectangles:', self.rectangles)
        for rectangle in self.rectangles:
            print(self.snapshot.coords(rectangle))

root = tk.Tk()
root.geometry('1400x680+50+50')
app = Application(master=root)
app.mainloop()
