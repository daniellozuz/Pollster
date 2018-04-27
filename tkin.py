#!/usr/bin/python3

import tkinter as tk
import cv2
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.template_path = tk.Entry(root)
        self.template_path.pack(side="top")

        self.cap = cv2.VideoCapture(0)
        self.live_footage = tk.Label(self.master)
        self.live_footage.pack(padx=10, pady=10)

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
            self.live_footage.imgtk = frame
            self.live_footage.config(image=frame)
        self.master.after(30, self.video_loop)

    def say_hi(self):
        print("hi there, everyone!")
        print(dir(self.template_path))

root = tk.Tk()
app = Application(master=root)
app.mainloop()
