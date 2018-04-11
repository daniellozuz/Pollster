import multiprocessing
import cv2
import time

STOP = 'STOP'

def cam_loop(pipe_parent):
    cap = cv2.VideoCapture(0)
    i = 0
    while True:
        _, img = cap.read()
        if img is not None:
            i += 1
            print('Read ' + str(i) + ' image')
            if i % 20 == 0:
                print('Added new img to the pipe.')
                pipe_parent.send(img)
        cv2.imshow('pepe', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            pipe_parent.send(STOP)
            break

def show_loop(pipe_child):
    cv2.namedWindow('pepe')
    while True:
        from_queue = pipe_child.recv()
        if from_queue == STOP:
            print("finished")
            break
        print("Processing image.")
        for i in range(10000000):
            continue
        print('Finished processing image.')

if __name__ == '__main__':

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    pipe_parent, pipe_child = multiprocessing.Pipe()

    cam_process = multiprocessing.Process(target=cam_loop, args=(pipe_parent, ))
    cam_process.start()

    show_process = multiprocessing.Process(target=show_loop, args=(pipe_child, ))
    show_process.start()

    cam_process.join()
    show_process.join()

    print(cam_process.is_alive())
    cv2.destroyAllWindows()
