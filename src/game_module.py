##TODO: wrapping using pygame

import time
import numpy as np

WINDOW_TIME = 1.0  # sec
CURR_THRESH = 0.3  # "norm"
WINDOW_THRESH = 0.9  # ratio

class Score:
    def __init__(self):
        self.final_score = 0.0
        self.curr_score = 0.0
        self.window_score = [0.0, 0.0]
        self.window_first_time = time.time()
        self.window_time = WINDOW_TIME
        self.curr_thresh = CURR_THRESH
        self.window_thresh = WINDOW_THRESH

    def reset_window(self):
        self.window_score = [0.0, 0.0]
        self.window_first_time = time.time()

    def update_scores(self, curr_score):
        self.curr_score = curr_score
        if curr_score < self.curr_thresh:
            self.window_score[0] += 1
        else:
            self.window_score[1] += 1
        if time.time() - self.window_first_time > WINDOW_TIME:
            if self.window_score[0] / np.sum(self.window_score) > self.window_thresh:
                self.final_score += 1
            self.window_score = [0.0, 0.0]
            self.window_first_time = time.time()


class GameModule:
    def __init__(self):
        self.state = 'INIT'
        self.scores = Score()

    def update_game(self, curr_score=np.nan):
        if self.state == 'INIT':
            self.state = 'PLAY'
            self.scores.reset_window()
        if not np.isnan(curr_score):
            self.scores.update_scores(curr_score)