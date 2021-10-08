import numpy as np
from stickers_module import StickersModule
import mediapipe as mp
import cv2

WINDOW_TIME = 1.0  # sec
CURR_THRESH = 0.3  # "norm"
WINDOW_THRESH = 0.5  # ratio
font = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)

import game_params  # TODO: read yaml / ini

class GameGUI:
    def __init__(self, wc_data=None, vid_data=None, width=None, height=None, multiply=2.0, stream_queue=None, png_path='./png/'):
        if width is None:
            self.screen_width = int(multiply * vid_data['width'])
            self.screen_height = int(multiply * vid_data['height'])
        else:
            self.screen_width = int(width)
            self.screen_height = int(height)
        self.wc_data = wc_data
        self.vid_data = vid_data
        self.mpDraw = mp.solutions.drawing_utils
        if self.screen_width == vid_data['width'] and self.screen_height == vid_data['height']:
            self.screen_vid_equal = True
        else:
            self.screen_vid_equal = False
        self.sticker_module = StickersModule(path=png_path)
        self.stream_queue = stream_queue
        print('GUI initialized -> {}X{}'.format(str(self.screen_width,), str(self.screen_height)))

    def choose_color(self, num, thresh=CURR_THRESH, low=GREEN, high=RED, opposite=False):
        if opposite:
            num, thresh = thresh, num
        if num < thresh:
            return low
        else:
            return high

    def merge_frames(self, final_frame, wc_frame, vid_frame):
        final_frame = cv2.addWeighted(final_frame, 0.25, vid_frame, 0.75, 0.0)
        wc_frame = cv2.resize(wc_frame, (int(self.screen_width/2.0), int(self.screen_height/2.0)))
        blended = final_frame[-wc_frame.shape[0]:, :wc_frame.shape[1]]
        blended[np.where(wc_frame > 0)] = wc_frame[np.where(wc_frame > 0)]
        return final_frame

    def update_gui(self, frame1, frame2, pose1, pose2, scores):
        if not self.screen_vid_equal:
            frame2 = cv2.resize(frame2, (self.screen_width, self.screen_height))

        gui_frame = np.zeros((self.screen_height, self.screen_width, 3), np.uint8)
        gui_frame = self.merge_frames(gui_frame, frame1, frame2)

        self.mpDraw.draw_landmarks(gui_frame, pose1, mp.solutions.pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=RED, thickness=2,
                                                                                           circle_radius=4))
        self.mpDraw.draw_landmarks(gui_frame, pose2, mp.solutions.pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=GREEN))


        gui_frame = scores.draw_scores(gui_frame)
        gui_frame = self.sticker_module.update_stickers(gui_frame, scores)

        cv2.putText(gui_frame, "SCORE: {:.0f}".format(scores.final_score),
                    (500, 100), font, 2, self.choose_color(scores.final_score, thresh=10, opposite=True), 4, cv2.LINE_AA)
        if self.stream_queue is None:
            cv2.imshow("compare", gui_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print('Exiting...')
                return False
        else:
            self.stream_queue.put(gui_frame)

        return True
