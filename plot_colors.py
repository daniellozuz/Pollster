import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_colors(video_path, hsv=False):
    capture = cv2.VideoCapture(video_path)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    means = []
    plt.axis([0, frame_count, 0, 256])
    while True:
        more, frame = capture.read()
        if not more:
            break
        if hsv:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean = [np.mean(frame[:, :, component]) for component in range(3)]
        means.append(mean)
    for component in range(3):
        plt.plot([mean[component] for mean in means], label=str(component))
    plt.legend()
    plt.show()
    capture.release()

plot_colors("ank/ank4.mp4", hsv=True)
