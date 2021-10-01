import time
import easygui
from pose_module import VideoPoseDetector
from video_module import PlaySyncVideo, VideoReader
from multiprocessing import Process, Queue, current_process
from game_module import GameModule
import subprocess

cmd = ['xrandr']
cmd2 = ['grep', '*']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
p.stdout.close()
resolution_string, junk = p2.communicate()
resolution = resolution_string.split()[0]
screen_width, screen_height = str(resolution).strip("b'").split('x')
screen_width = int(screen_width)
screen_height = int(screen_height)

def get_video_data(file_path=None, video_reader=None):
    if file_path:
        video_reader = VideoReader(file_path)
    return video_reader.get_data()

def web_cam_process(res_queue, web_cam_data):
    file_path = 0
    vid_detector = VideoPoseDetector(file_path, flip_h=True, MODEL_COMPLEXITY=0, ENABLE_SEGMENTATION=True)
    web_cam_data.put(get_video_data(video_reader=vid_detector.vid_reader))
    vid_detector.create_pose_data(output='', visualize=False, res_queue=res_queue)

def video_process(res_queue, video_data):
    file_path = easygui.fileopenbox(default='./curr_video/*.mp4')
    video_data.put(get_video_data(file_path=file_path))
    res = PlaySyncVideo(video_path=file_path, pose_path='same', draw_lm=False, res_queue=res_queue, local_video=True, visualize=False)
    next(res)

def game_process(res_queue1, res_queue2, web_cam_data, video_data):
    game_module = GameModule(wc_data=web_cam_data.get(),
                             vid_data=video_data.get()
                             , width=screen_width*0.9, height=screen_height*0.9)
    while True:
        last_value1, last_value2 = None, None
        while res_queue1.qsize() > 0:
            last_value1 = res_queue1.get()
        while res_queue2.qsize() > 0:
            last_value2 = res_queue2.get()
        game_module.update_game(last_value1, last_value2)
        while res_queue1.qsize() == 0 or res_queue2.qsize() == 0:
            time.sleep(0.01)

if __name__ == '__main__':
    wc_queue = Queue()
    wc_data = Queue()
    wc_process = Process(name='web_cam_process', target=web_cam_process, args=(wc_queue, wc_data))
    wc_process.daemon = True
    wc_process.start()

    vid_queue = Queue()
    video_data = Queue()
    vid_process = Process(name='video_process', target=video_process, args=(vid_queue, video_data))
    vid_process.daemon = True
    vid_process.start()

    cp_process = Process(name='compare_poses', target=game_process, args=(wc_queue, vid_queue, wc_data, video_data))
    cp_process.daemon = True
    cp_process.start()

    while True:
        time.sleep(10000)
