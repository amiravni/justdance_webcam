import cv2
import numpy as np
import mediapipe as mp


def show_two_poses(pose1, pose2):
    blank_image = np.zeros((1000, 1000, 3), np.uint8)
    mpDraw = mp.solutions.drawing_utils
    mpDraw.draw_landmarks(blank_image, pose1, mp.solutions.pose.POSE_CONNECTIONS,
                          landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255)))
    mpDraw.draw_landmarks(blank_image, pose2, mp.solutions.pose.POSE_CONNECTIONS,
                          landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0)))
    cv2.imshow("compare", blank_image)
    cv2.waitKey(1)

