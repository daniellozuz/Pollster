import multiprocessing
import cv2
import time
from itertools import count


WEBCAM = 0
# Not possible to set those values for webcam
CAMERA_SETTINGS = {
    #cv2.CAP_PROP_BRIGHTNESS: 151,
    #cv2.CAP_PROP_CONTRAST: 43,
    #cv2.CAP_PROP_SATURATION: 12,
    #cv2.CAP_PROP_HUE: 13,
    #cv2.CAP_PROP_EXPOSURE: -5,
}


class ImageProducer(object):
    def __init__(self, input_stream, camera_settings):
        self.cap = cv2.VideoCapture(input_stream)
        self.setup_camera(camera_settings)

    def setup_camera(self, camera_settings):
        for option, value in camera_settings.items():
            self.cap.set(option, value)

    def frame_generator(self, step=1, show_every_frame=False):
        index = count()
        while True:
            more, image = self.cap.read()
            if not more:
                return
            if show_every_frame:
                cv2.imshow('Live display', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
            i = next(index)
            if i % step == 0:
                yield i, image

    def produce(self, pipe_parent):
        for index, image in self.frame_generator(step=2, show_every_frame=True):
            if index % 20 == 0: # Decide when to snapshot (poll recognition)
                pipe_parent.send(image)
                print('SENT image.')
        self.cap.release()
        pipe_parent.send('STOP')


class ImageConsumer(object):
    def consume(self, pipe_child):
        while True:
            image = pipe_child.recv()
            if image == 'STOP':
                print("finished")
                break
            print("RECEIVED image, processing...")
            cv2.imshow('Processing results', image)
            cv2.waitKey(1)
            for _ in range(100000):
                continue
            print('Finished processing image.')


class Pollster(object):
    def run(self, input_stream, camera_settings):
        pipe_parent, pipe_child = multiprocessing.Pipe()

        producer = ImageProducer(input_stream, camera_settings)
        consumer = ImageConsumer()

        producer_process = multiprocessing.Process(target=producer.produce, args=(pipe_child, ))
        consumer_process = multiprocessing.Process(target=consumer.consume, args=(pipe_parent, ))

        producer_process.start()
        consumer_process.start()

        consumer_process.join()
        producer_process.join()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    Pollster().run(input_stream=WEBCAM, camera_settings=CAMERA_SETTINGS)