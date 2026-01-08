import av
import time
import numpy as np 
from djitellopy import Tello
from threading import Thread, Lock


VIDEO_UDP_PORT = 11111
VIDEO_URL = f"udp://192.168.10.1:{VIDEO_UDP_PORT}"


class TelloVideoGrabber:
    """
    This class read frames using PyAV in background. Use
    backgroundFrameRead.frame to get the current frame.
    """

    def __init__(self):
        self.lock = Lock()
        self._frame = np.zeros([300, 400, 3], dtype=np.uint8)
        self.connected = False
        self.container = None
        self.loop = False

        self.init_container()

    def start(self):
        """Start the frame update worker
        Internal method, you normally wouldn't call this yourself.
        """
        self.worker = Thread(target=self.update_frame, args=(), daemon=True)
        self.loop = True
        self.worker.start()
    
    def init_container(self):
        tries = 0
        while self.connected is not True and tries < 5:
            print('[CAMERA] Trying to grab video frames...')
            if self.container:
                self.container.close()
            try:
                self.container = av.open(VIDEO_URL, timeout=(Tello.FRAME_GRAB_TIMEOUT, None))
                self.connected = True
            except Exception as err:
                self.connected = False
                print('[CAMERA] Failed to grab video frames from video stream')
                print(err)
            finally: 
                tries += 1
            
            time.sleep(1)
    
    def stop_container(self):
        if self.container is not None:
            self.container.close()
        self.container = None
        self.connected = False

    def stop(self):
        self.loop = False

        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2)
        
        self.stop_container()
        print("[CAMERA] Thread is not alive anymore")

    def update_frame(self):
        """Thread worker function to retrieve frames using PyAV
        Internal method, you normally wouldn't call this yourself.
        """
        while self.loop:
            try:
                if self.connected:
                    for frame in self.container.decode(video=0):
                        self.frame = np.array(frame.to_image())
                        if not self.loop:
                            break

            except av.error.ExitError:
                print('[CAMERA][DECODER] Do not have enough frames for decoding, please try again or increase video fps before get_frame_read()')
                self.stop_container()
                self.init_container()
        
    @property
    def frame(self):
        """
        Access the frame variable directly
        """
        with self.lock:
            return self._frame

    @frame.setter
    def frame(self, value):
        """
        Define the frame variable with the lock
        """
        with self.lock:
            self._frame = value