import multiprocessing
import cv2
import time


CAMERA_SETTINGS = {
    cv2.CAP_PROP_BRIGHTNESS: 10,
    cv2.CAP_PROP_CONTRAST: 11,
    cv2.CAP_PROP_SATURATION: 12,
    cv2.CAP_PROP_HUE: 13,
    cv2.CAP_PROP_EXPOSURE: 15,
}

class Pollster(object):

    def __init__(self, input_stream, camera_settings):
        self.input_stream = input_stream
        self.camera_settings = camera_settings
        self.cap = None
        self.stop = 'STOP'

    def setup_camera(self):
        for option, value in self.camera_settings.items():
            self.cap.set(option, value)

    def open_stream(self):
        self.cap = cv2.VideoCapture(self.input_stream)

    def cam_loop(self, pipe_parent):
        i = 0
        while True:
            _, img = self.cap.read()
            if img is not None:
                i += 1
                print('Read ' + str(i) + ' image')
                if i % 20 == 0:
                    print('Added new img to the pipe.')
                    print(type(img))
                    pipe_parent.send('x')
            cv2.imshow('pepe', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                pipe_parent.send(self.stop)
                break

    def show_loop(self, pipe_child):
        cv2.namedWindow('pepe')
        while True:
            from_queue = pipe_child.recv()
            if from_queue == self.stop:
                print("finished")
                break
            print("Processing image.")
            for i in range(10000000):
                continue
            print('Finished processing image.')

    def run(self):
        self.open_stream()
        self.setup_camera()

        #logger = multiprocessing.log_to_stderr()
        #logger.setLevel(multiprocessing.SUBDEBUG)

        pipe_parent, pipe_child = multiprocessing.Pipe(duplex=False)

        cam_process = multiprocessing.Process(target=self.cam_loop, args=(pipe_parent, ))
        cam_process.start()

        show_process = multiprocessing.Process(target=self.show_loop, args=(pipe_child, ))
        show_process.start()

        cam_process.join()
        show_process.join()

        print(cam_process.is_alive())
        cv2.destroyAllWindows()

class ImageProducer(object):
    dupa = cv2.VideoCapture(0)
    #cap = None
    def __init__(self, input_stream, camera_settings):
        self.input_stream = input_stream
        self.camera_settings = camera_settings
        cap = cv2.VideoCapture(self.input_stream) # THIS NEEDS TO BE PICKLABLE IN ORDER TO USE PIPES ON WINDOWS XD
        dupa = self.input_stream
        
        # self.setup_camera()

    # def setup_camera(self):
    #     for option, value in self.camera_settings.items():
    #         self.cap.set(option, value)
    @classmethod
    def produce(cls, pipe_parent):
        print(cls.dupa)
        for i in range(10):
            time.sleep(1)
            pipe_parent.send(1)
        pipe_parent.send('STOP')
        

class ImageConsumer(object):
    # def __init__(self, pipe_child):
    #     self.pipe_child = pipe_child
    
    def consume(self, pipe_child):
        while True:
            from_queue = pipe_child.recv()
            if from_queue == 'STOP':
                print("finished")
                break
            print("Processing image.")
            for i in range(10000):
                continue
            print('Finished processing image.')

if __name__ == '__main__':

    pipe_parent, pipe_child = multiprocessing.Pipe()

    p = ImageProducer
    c = ImageConsumer()

    print(type(p.produce))

    p_process = multiprocessing.Process(target=p.produce, args=(pipe_child, ))
    p_process.start()

    c_process = multiprocessing.Process(target=c.consume, args=(pipe_parent, ))
    c_process.start()

    c_process.join()
    p_process.join()

    print(p_process.is_alive())



# STOP = 'STOP'

# def cam_loop(pipe_parent):
#     cap = cv2.VideoCapture(0)
#     i = 0
#     while True:
#         _, img = cap.read()
#         if img is not None:
#             i += 1
#             print('Read ' + str(i) + ' image')
#             if i % 20 == 0:
#                 print('Added new img to the pipe.')
#                 pipe_parent.send(img)
#         cv2.imshow('pepe', img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cap.release()
#             pipe_parent.send(STOP)
#             break

# def show_loop(pipe_child):
#     cv2.namedWindow('pepe')
#     while True:
#         from_queue = pipe_child.recv()
#         if from_queue == STOP:
#             print("finished")
#             break
#         print("Processing image.")
#         for i in range(10000000):
#             continue
#         print('Finished processing image.')

# if __name__ == '__main__':

#     logger = multiprocessing.log_to_stderr()
#     logger.setLevel(multiprocessing.SUBDEBUG)

#     pipe_parent, pipe_child = multiprocessing.Pipe()

#     cam_process = multiprocessing.Process(target=cam_loop, args=(pipe_parent, ))
#     cam_process.start()

#     show_process = multiprocessing.Process(target=show_loop, args=(pipe_child, ))
#     show_process.start()

#     cam_process.join()
#     show_process.join()

#     print(cam_process.is_alive())
#     cv2.destroyAllWindows()
