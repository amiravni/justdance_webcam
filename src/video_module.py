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


def gen_frame_to_web(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def PlaySyncVideo(video_path, pose_path, draw_lm=False, res_queue=None, local_video=True):
    if pose_path == 'same':
        pose_path = video_path + '.pkl'
    with open(pose_path, 'rb') as handle:
        pose_data = pickle.load(handle)
    mpDraw = mp.solutions.drawing_utils
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    h, w = video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    if h < 400.0 or w < 400:
        ratio = h/w
        new_width = 1440
        cv2.resizeWindow('Video', new_width, int(ratio * new_width))
    player = MediaPlayer(video_path)
    frame_count = 0
    last_frame_time = 0.0
    first_time = time.time()
    while True:
        if (time.time() - first_time) >= last_frame_time + (1 / fps):
            grabbed, frame = video.read()
            if draw_lm:
                mpDraw.draw_landmarks(frame, pose_data[frame_count][0], mp.solutions.pose.POSE_CONNECTIONS)
            if res_queue:
                res_queue.put(pose_data[frame_count])
            audio_frame, val = player.get_frame()
            last_frame_time = val
            frame_count += 1
            if not grabbed:
                print("End of video")
                break
            if local_video:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                cv2.imshow("Video", frame)
            else:
                yield gen_frame_to_web(frame)
        else:
            if local_video:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    video.release()
    cv2.destroyAllWindows()


class VideoReader:
    def __init__(self, path, show=False, start_frame=0, start_sec=0):
        self.cap = cv2.VideoCapture(path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.show = show
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if start_sec > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.fps * start_sec)
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
                 ref_video=None,  ## send here "cap" to get all attributes
                 fps=30,
                 frame_width=1920,
                 frame_height=1080,
                 show=False):

        if ref_video is not None:
            frame_width = ref_video.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = ref_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if ref_video.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                fps = ref_video.get(cv2.CAP_PROP_FPS)
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


if __name__ == '__main__':
    res = PlaySyncVideo(video_path='./curr_video/Just Dance 2016 - Good Feeling - Flo rida - 5 Stars.mp4',
                  pose_path='same')
    next(res)

    if False:
        # video_stream = VideoReader('/home/makeruser/nitay/video.mp4', show=True)
        video_stream = VideoReader(0, show=True)
        video_file = VideoWriter('../videos/test_video.mp4',
                                 ref_video=video_stream.cap)
        tmp_frame = []
        while (tmp_frame is not None):
            tmp_frame = video_stream.read_frame()
            video_file.write_frame(tmp_frame)
        video_file.close()
