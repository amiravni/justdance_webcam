import time

from pose_module import VideoPoseDetector
from video_module import PlaySyncVideo
from multiprocessing import Process, Queue, current_process
from analysis_module import compare_pose

def web_cam_process(res_queue):
    file_path = 0
    vid_detector = VideoPoseDetector(file_path, flip_h=True)
    vid_detector.create_pose_data(output='', visualize=True, res_queue=res_queue)

def video_process(res_queue):
    file_path = './curr_video/Just Dance 2016 - Good Feeling - Flo rida - 5 Stars.mp4'
    PlaySyncVideo(video_path=file_path, pose_path='same', res_queue=res_queue)

def compare_poses(res_queue1, res_queue2):
    while True:
        last_value1, last_value2 = None, None
        while res_queue1.qsize() > 0:
            last_value1 = res_queue1.get()
        while res_queue2.qsize() > 0:
            last_value2 = res_queue2.get()
        compare_pose(last_value1, last_value2)
        while res_queue1.qsize() == 0 or res_queue2.qsize() == 0:
            time.sleep(0.01)

wc_queue = Queue()
wc_process = Process(name='web_cam_process', target=web_cam_process, args=(wc_queue,))
wc_process.daemon = True
wc_process.start()

vid_queue = Queue()
vid_process = Process(name='video_process', target=video_process, args=(vid_queue,))
vid_process.daemon = True
vid_process.start()

cp_process = Process(name='compare_poses', target=compare_poses, args=(wc_queue, vid_queue))
cp_process.daemon = True
cp_process.start()

while True:
    time.sleep(10000)