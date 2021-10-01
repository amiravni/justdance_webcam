import cv2
import mediapipe as mp
import time
import pickle
from video_module import VideoReader, VideoWriter
import numpy as np


class PoseDetector:

    def __init__(self, STATIC_IMAGE_MODE=False,
                 MODEL_COMPLEXITY=1,
                 SMOOTH_LANDMARKS=False,
                 ENABLE_SEGMENTATION=False,
                 SMOOTH_SEGMENTATION=True,
                 MIN_DETECTION_CONFIDENCE = 0.5,
                 MIN_TRACKING_CONFIDENCE = 0.5):


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=STATIC_IMAGE_MODE,
                                     model_complexity=MODEL_COMPLEXITY,
                                     smooth_landmarks=SMOOTH_LANDMARKS,
                                     enable_segmentation=ENABLE_SEGMENTATION,
                                     smooth_segmentation=SMOOTH_SEGMENTATION,
                                     min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                                     min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                cz_est = int(lm.z * w)
                #lmList.append([id, cx, cy, cz_est])
                lmList.append([id, lm.x, lm.y, lm.z])
                if draw:
                    try:
                        if lm.visibility > 0.5:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        else:
                            cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
                    except:
                        print(lm.z)
        return lmList


class VideoPoseDetector(PoseDetector):
    def __init__(self, video_path, flip_h=False, start_sec=0, **kwargs):
        PoseDetector.__init__(self, **kwargs)
        self.video_path = video_path
        self.flip_h = flip_h
        self.vid_reader = VideoReader(video_path, start_sec=start_sec)

        self.use_mask = ('ENABLE_SEGMENTATION' in kwargs and kwargs['ENABLE_SEGMENTATION'])
        if self.use_mask:
            self.bg_image = np.zeros((int(self.vid_reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                      int(self.vid_reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
            self.bg_image[:] = (0, 0, 0)

    def create_pose_data(self, output='', visualize=False, save_video=False, res_queue=None, print_progress=False):
        if save_video:
            self.video_path = self.video_path + '_new.mp4'
            video_writer = VideoWriter(path=self.video_path, ref_video=self.vid_reader.cap)
        if output == 'same':
            output = self.video_path + '.pkl'
        pose_data = []
        frame_count = 0
        pTime = 0

        while True:
            img = self.vid_reader.read_frame(self.flip_h)
            #img = cv2.resize(img, (360, 640))
            if img is None:
                break
            img = self.findPose(img, draw=visualize)

            if self.use_mask and self.results.segmentation_mask is not None:
                try:
                    condition = np.stack((self.results.segmentation_mask,) * 3, axis=-1) > 0.1
                    img = np.where(condition, img, self.bg_image)
                except:
                    pass
                    #print('segmentation failed')


            if save_video:
                video_writer.write_frame(img)
            if len(output) > 0:
                pose_data.append([self.results.pose_landmarks, self.results.pose_world_landmarks])
            if res_queue:
                res_queue.put([[self.results.pose_landmarks, self.results.pose_world_landmarks], img])
            frame_count += 1
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            #print(fps)
            if visualize:
                cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                cv2.imshow("Image", img)
                cv2.waitKey(1)
            else:
                if print_progress and not frame_count % 30:
                    print(fps, (frame_count / self.vid_reader.frame_count) * 100.0)

        if len(output) > 0:
            with open(output, 'wb') as fid:
                pickle.dump(pose_data, fid)


def main():
    #file_path = './curr_video/Just Dance 2016 - Good Feeling - Flo rida - 5 Stars.mp4'
    file_path = 0
    vid_detector = VideoPoseDetector(file_path, MODEL_COMPLEXITY=0, ENABLE_SEGMENTATION=True)
    vid_detector.create_pose_data(output='', visualize=True)

if __name__ == "__main__":
    main()