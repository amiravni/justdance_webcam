##TODO: wrapping using pygame

import time
import numpy as np
import cv2
from analysis_module import compare_pose, get_bias_var, normalize_landmark
import mediapipe as mp
import glob

DEBUG = False

WINDOW_TIME = 1.0  # sec
CURR_THRESH = 0.3  # "norm"
WINDOW_THRESH = 0.5  # ratio

font = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)

pose_idxs = {
    'BODY': [11, 12, 23, 24],
    'LEFT_HAND': [13, 15, 17, 19, 21],
    'RIGHT_HAND': [14, 16, 18, 20, 22],
    'LEFT_FOOT': [25, 27, 29, 31],
    'RIGHT_FOOT': [26, 28, 30, 32],
    'HEAD': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # not in use
}

class Sticker:
    def __init__(self, img):  # TODO: add more inputs
        self.img = img[:, :, :3]
        self.mask = img[:, :, -1]
        self.start_time = time.time()
        self.fade_in = 0.2
        self.full = 0.2
        self.fade_out = 1.0
        self.total_time = self.fade_in + self.full + self.fade_out
        self.loc_w = 900
        self.loc_h = 100
        self.angle = np.random.randint(20) - 10  # TODO: use this
        self.move_w = np.random.randint(60) - 30
        self.move_h = np.random.randint(60) - 30
        self.done = False

    def add_to_image(self, img):
        time_from_start = time.time() - self.start_time
        ratio = time_from_start / self.total_time
        if time_from_start < self.fade_in:
            beta = time_from_start / self.fade_in
            alpha = 1 - beta
        elif time_from_start < self.fade_in + self.full:
            beta = 1
            alpha = 1 - beta
        elif time_from_start < self.fade_in + self.full + self.fade_out:
            alpha = (time_from_start - (self.fade_in + self.full)) / self.fade_out
            beta = 1 - alpha
        else:
            self.done = True

        if not self.done:
            move_h = int(ratio * self.move_h)
            move_w = int(ratio * self.move_w)
            location = [self.loc_h + move_h, self.loc_h + self.img.shape[0] + move_h,
                        self.loc_w + move_w, self.loc_w + self.img.shape[1] + move_w]
            blend = img[location[0]:location[1], location[2]:location[3]]
            mask = np.where(self.mask)
            blend[mask] = cv2.addWeighted(blend[mask], alpha, self.img[mask], beta, 0.0).squeeze()
            # img[location[0]:location[1], location[2]:location[3]] =\
            #     cv2.addWeighted(img[location[0]:location[1], location[2]:location[3]],
            #                     alpha, self.img, beta, 0.0)
        return img

class StickerModule:
    def __init__(self, path):
        stickers_path = glob.glob(path + '*.png')
        self.stickers = []
        for sticker_path in stickers_path:
            img = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
            self.stickers.append(cv2.resize(img, (200, int(img.shape[1]/img.shape[0]*200))))
        self.sticker_sequence = []
        self.next_sticker_score = 3

    def update_stickers(self, img, scores):
        if scores.final_score > self.next_sticker_score:
            self.add_sticker()
            self.next_sticker_score += 3
        self.draw_sequence(img)
        return img


    def add_sticker(self):
        img = self.stickers[np.random.randint(len(self.stickers))]
        self.sticker_sequence.append(Sticker(img=img))

    def draw_sequence(self, img):
        for sticker in self.sticker_sequence:
            if not sticker.done:
                img = sticker.add_to_image(img)
            if sticker.done:
                self.sticker_sequence.remove(sticker)
        return img





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
        self.sticker_module = StickerModule(path=png_path)
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


class Score:
    def __init__(self):
        self.final_score = 0.0
        self.curr_score = 0.0
        self.vec_dist = []
        self.window_score = [0.0, 0.0]
        self.window_score_ratio = [0, 0]
        self.time_passed = 0.0
        self.window_first_time = time.time()
        self.window_time = WINDOW_TIME
        self.curr_thresh = CURR_THRESH
        self.window_thresh = WINDOW_THRESH

    def reset_window(self):
        self.window_score = [0.0, 0.0]
        self.window_score_ratio = [0, 0]
        self.window_first_time = time.time()

    def update_scores(self, curr_score, vec_dist=[]):
        self.curr_score = curr_score
        self.vec_dist = vec_dist
        if curr_score < self.curr_thresh:
            self.window_score[0] += 1
        else:
            self.window_score[1] += 1
        sum_cnt = np.sum(self.window_score)
        self.time_passed = time.time() - self.window_first_time
        if sum_cnt > 0:
            self.window_score_ratio = [self.window_score[0] / sum_cnt,
                                       self.window_score[1] / sum_cnt]
        else:
            self.window_score_ratio = [0, 0]

        if self.time_passed > WINDOW_TIME:
            if self.window_score[0] / np.sum(self.window_score) > self.window_thresh:
                self.final_score += 1
            self.window_score = [0.0, 0.0]
            self.window_first_time = time.time()

    def draw_scores(self, gui_frame, x1=(650, 150), x2=(700, 250)):
        if DEBUG:
            if self.curr_score:
                cv2.putText(gui_frame, "{:.2f}".format(self.curr_score), (100, 100),
                            font, 3, self.choose_color(self.curr_score), 5, cv2.LINE_AA)
            if self.vec_dist:
                gap = 0
                for dist in self.vec_dist:
                    cv2.putText(gui_frame, "{:.2f}".format(dist), (100 + gap, 200), font, 2, self.choose_color(dist), 4,
                                cv2.LINE_AA)
                    gap += 200

        height = x2[1] - x1[1]
        time_ratio = self.time_passed / self.window_time
        green_box_sz = int(height * time_ratio * (1-self.window_score_ratio[0]))
        red_box_sz = int(height * time_ratio * (1-self.window_score_ratio[1]))
        black_box_sz = int(height * (1-time_ratio))
        gui_frame = cv2.rectangle(gui_frame, x1, x2, BLUE, 1)
        gui_frame = cv2.rectangle(gui_frame, (x1[0], x1[1]+red_box_sz), (x2[0], x2[1]), RED, -1)
        gui_frame = cv2.rectangle(gui_frame, (x1[0], x1[1]+black_box_sz), (x2[0], x2[1]-green_box_sz), GREEN, -1)
        gui_frame = cv2.rectangle(gui_frame, (x1[0], x1[1]), (x2[0], x1[1]+black_box_sz), BLACK, -1)
        return gui_frame

class GameModule:
    def __init__(self, wc_data, vid_data, width=None, height=None, stream_queue=None, png_path='./png/'):
        self.state = 'INIT'
        self.scores = Score()
        self.game_gui = GameGUI(wc_data=wc_data, vid_data=vid_data, width=width, height=height, stream_queue=stream_queue, png_path=png_path)
        print('game initialized')


    def extract_data(self, last_value1, last_value2):
        pose1, pose2, frame1, frame2 = None, None, None, None
        if last_value1 and last_value2:
            pose1 = last_value1[0].copy()
            pose2 = last_value2[0].copy()
            frame1 = last_value1[1].copy()
            frame2 = last_value2[1].copy()
        return pose1, pose2, frame1, frame2

    def update_game(self, last_value1, last_value2):
        pose1, pose2, frame1, frame2 = self.extract_data(last_value1, last_value2)
        curr_score, vec_dist, pose1_2d_norm, pose2_2d_norm = self.compare_poses(pose1, pose2)
        if pose1_2d_norm is not None and pose2_2d_norm is not None:
            if self.state == 'INIT':
                self.state = 'PLAY'
                self.scores.reset_window()
            if curr_score and not np.isnan(curr_score):
                self.scores.update_scores(curr_score, vec_dist=vec_dist)
            if not self.game_gui.update_gui(frame1, frame2, pose1_2d_norm, pose2_2d_norm, self.scores):
                self.state = 'END'
                return False
        return True

    def compare_poses(self, last_value1, last_value2):
        return compare_pose(last_value1, last_value2)
