import cv2 as cv
from .utils import scale_image

class VideoReader():
    """Reads the frames of a video"""
    def __init__(self, cap:cv.VideoCapture, scale:float=1, return_original:bool=False):
        """Read all the frames of a video in RAM memory

        Keyword arguments:
        cap -- the Video Capture object of the video
        scale -- the scale to which the frames of the video are scaled (default 1)
        return_original -- returns also the frames with their original size (default False)
        """
        self.scale = scale
        self.return_original = return_original
        self.read_video_frames(cap)
        self.n_frames = len(self.frames)
        self.current = 0
        self.frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    def read_video_frames(self, cap:cv.VideoCapture):
        """Reads all the frames of the video to memory.

        This approach speeds up the reading process.
        """
        frames = []
        original_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.return_original:
                original_frames.append(frame.copy())
            frames.append(scale_image(frame, self.scale))
        self.frames = frames
        if self.return_original:
            self.original_frames = original_frames

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.n_frames:
            frame = self.frames[self.current]
            if self.return_original:
                original_frame = self.original_frames[self.current]
            self.current += 1
            if self.return_original:
                return frame, original_frame
            else:
                return frame
        else:
            raise StopIteration
