import cv2
import time
import numpy as np

DEBUG = False
WINDOW_TIME = 1.0  # sec
CURR_THRESH = 0.3  # "norm"
WINDOW_THRESH = 0.5  # ratio
font = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)

import game_params  # TODO: read yaml / ini

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
