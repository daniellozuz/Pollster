import multiprocessing
import cv2
import time
from itertools import count
import numpy as np
from collections import deque

WEBCAM = 0
CAMERA_SETTINGS = {
    #cv2.CAP_PROP_BRIGHTNESS: 151,
    #cv2.CAP_PROP_CONTRAST: 43,
    #cv2.CAP_PROP_SATURATION: 12,
    #cv2.CAP_PROP_HUE: 13,
    #cv2.CAP_PROP_EXPOSURE: -5,
}

class PollProducer(object):
    WAITING_FOR_EMPTY_CHAMBER = 0
    WAITING_FOR_SNAPSHOT = 1

    DEQUE_LENGTH = 10
    MAX_VARIANCE = 1
    LOWER_BRIGHTNESS = 1
    UPPER_BRIGHTNESS = 50
    FRAME_STEP = 1

    def __init__(self, input_stream, camera_settings):
        self.cap = cv2.VideoCapture(input_stream)
        self._setup_camera(camera_settings)

    def _setup_camera(self, camera_settings):
        for option, value in camera_settings.items():
            self.cap.set(option, value)

    def _frame_generator(self, step=1, live_display=False):
        frame_counter = count()
        while True:
            more, frame = self.cap.read()
            time.sleep(0.03)
            if not more:
                return
            if live_display:
                cv2.imshow('Live display', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
            index = next(frame_counter)
            if index % step == 0:
                yield index, frame

    def send_polls(self, pipe_parent):
        state = PollProducer.WAITING_FOR_SNAPSHOT
        v_means = deque(maxlen=PollProducer.DEQUE_LENGTH)
        for index, frame in self._frame_generator(step=PollProducer.FRAME_STEP, live_display=True):
            v_mean = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,2])
            v_means.append(v_mean)
            if all([state == PollProducer.WAITING_FOR_SNAPSHOT,
                    v_mean > PollProducer.UPPER_BRIGHTNESS,
                    np.var(v_means) < PollProducer.MAX_VARIANCE]):
                pipe_parent.send(frame)
                print('SENT frame.')
                v_means.clear()
                state = PollProducer.WAITING_FOR_EMPTY_CHAMBER
            if all([state == PollProducer.WAITING_FOR_EMPTY_CHAMBER,
                    v_mean < PollProducer.LOWER_BRIGHTNESS]):
                state = PollProducer.WAITING_FOR_SNAPSHOT
        self.cap.release()
        pipe_parent.send('STOP')


class PollConsumer(object):
    def receive_polls(self, pipe_child):
        while True:
            frame = pipe_child.recv()
            if frame == 'STOP':
                print("finished")
                break
            print("RECEIVED a poll, processing...")
            for _ in range(100000):
                continue
            cv2.imshow('Processed poll', frame)
            cv2.waitKey(1)
            print('Finished processing a poll.')


class Pollster(object):
    def run(self, input_stream, camera_settings):
        pipe_parent, pipe_child = multiprocessing.Pipe()

        producer = PollProducer(input_stream, camera_settings)
        consumer = PollConsumer()

        producer_process = multiprocessing.Process(target=producer.send_polls,
                                                   args=(pipe_child, ))
        consumer_process = multiprocessing.Process(target=consumer.receive_polls,
                                                   args=(pipe_parent, ))

        producer_process.start()
        consumer_process.start()

        consumer_process.join()
        producer_process.join()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    Pollster().run(input_stream='src_video/after_cam_seq.avi',
                   camera_settings=CAMERA_SETTINGS)
