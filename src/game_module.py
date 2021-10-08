##TODO: wrapping using pygame

import numpy as np
from algo.analysis_module import compare_pose
from game.score import Score
from game.gui import GameGUI

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
