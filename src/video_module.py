import cv2
from ffpyplayer.player import MediaPlayer
import numpy as np
import time
import mediapipe as mp
import pickle

def is_open(cap):
  if (cap.isOpened() == False):
    print("Error opening video stream or file")
    return False
  return True

def PlaySyncVideo(video_path, pose_path, res_queue=None):
    if pose_path == 'same':
        pose_path = video_path + '.pkl'
    with open(pose_path, 'rb') as handle:
        pose_data = pickle.load(handle)
    mpDraw = mp.solutions.drawing_utils
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    player = MediaPlayer(video_path)
    frame_count = 0
    last_frame_time = 0.0
    first_time = time.time()
    while True:
        if (time.time() - first_time) >= last_frame_time + (1 / fps):
            grabbed, frame = video.read()
            mpDraw.draw_landmarks(frame, pose_data[frame_count], mp.solutions.pose.POSE_CONNECTIONS)
            if res_queue:
                res_queue.put(pose_data[frame_count])
            audio_frame, val = player.get_frame()
            last_frame_time = val
            frame_count += 1
            if not grabbed:
                print("End of video")
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.imshow("Video", frame)
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video.release()
    cv2.destroyAllWindows()

class VideoReader:
    def __init__(self, path, show=False, start_frame=0):
        self.cap = cv2.VideoCapture(path)
        self.show = show
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if not is_open(self.cap):
            return None

    def read_frame(self, flip_h=False):
        if not is_open(self.cap):
            return None
        ret, frame = self.cap.read()
        if ret:
            if flip_h:
                frame = cv2.flip(frame, 1)
            if self.show:
                # Display the resulting frame
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                  self.close()
                  return None
            return frame
        else:
            print('No Frame')
            self.close()
            return None

    def close(self):
        self.cap.release()
        if self.show:
            # Closes all the frames
            cv2.destroyAllWindows()


class VideoWriter:
  def __init__(self, path,
               vid_type=cv2.VideoWriter_fourcc(*'mp4v'),
               ref_video=None, ## send here "cap" to get all attributes
               fps=30,
               frame_width=1920,
               frame_height=1080,
               show=False):

    if ref_video is not None:
      frame_width = ref_video.get(cv2.CAP_PROP_FRAME_WIDTH)
      frame_height = ref_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
      if ref_video.get(cv2.CAP_PROP_POS_FRAMES) > 0:
        fps = ref_video.get(cv2.CAP_PROP_POS_FRAMES)
      ## vid type = ref_video.get(cv2.CAP_PROP_FOURCC)????
    self.out = cv2.VideoWriter(path, vid_type, fps, (int(frame_width), int(frame_height)))
    self.show = show

  def write_frame(self, frame):
      if self.show:
          # Display the resulting frame
          cv2.imshow('Frame_write', frame)
          if cv2.waitKey(25) & 0xFF == ord('q'):
              self.close()
              return None
      self.out.write(frame)

  def close(self):
    self.out.release()


if __name__=='__main__':
  PlaySyncVideo(video_path='./curr_video/Just Dance 2016 - Good Feeling - Flo rida - 5 Stars.mp4',
                pose_path='same')

  if False:
      #video_stream = VideoReader('/home/makeruser/nitay/video.mp4', show=True)
      video_stream = VideoReader(0, show=True)
      video_file = VideoWriter('../videos/test_video.mp4',
                               ref_video=video_stream.cap)
      tmp_frame = []
      while(tmp_frame is not None):
        tmp_frame = video_stream.read_frame()
        video_file.write_frame(tmp_frame)
      video_file.close()