import time
import easygui
from algo.pose_module import VideoPoseDetector
from video.video_module import PlaySyncVideo, VideoReader
from multiprocessing import Process, Queue
from game_module import GameModule
from screeninfo import get_monitors


for m in get_monitors():
    screen_width = m.width
    screen_height = m.height
    break


def get_video_data(file_path=None, video_reader=None):
    if file_path:
        video_reader = VideoReader(file_path)
    return video_reader.get_data()

def web_cam_process(res_queue, web_cam_data):
    cam_location = 0
    vid_detector = VideoPoseDetector(cam_location, flip_h=True, MODEL_COMPLEXITY=0, ENABLE_SEGMENTATION=True)
    if not vid_detector.vid_reader.cap.isOpened():
        vid_detector = VideoPoseDetector(cam_location + 1, flip_h=True, MODEL_COMPLEXITY=0, ENABLE_SEGMENTATION=True)
    web_cam_data.put(get_video_data(video_reader=vid_detector.vid_reader))
    vid_detector.create_pose_data(output='', visualize=False, res_queue=res_queue)


def video_process(res_queue, video_data, default_path='./curr_video/*.mp4'):
    file_path = easygui.fileopenbox(default=default_path)
    video_data.put(get_video_data(file_path=file_path))
    res = PlaySyncVideo(video_path=file_path, pose_path='same', draw_lm=False, res_queue=res_queue, visualize=False)
    #next(res)

def game_process(res_queue1, res_queue2, web_cam_data, video_data, stream_queue=None, png_path='./png/'):
    game_module = GameModule(wc_data=web_cam_data.get(),
                             vid_data=video_data.get(),
                             width=screen_width*0.9,
                             height=screen_height*0.9,
                             stream_queue=stream_queue,
                             png_path=png_path)
    while True:
        last_value1, last_value2 = None, None
        while res_queue1.qsize() > 0:
            last_value1 = res_queue1.get()
        while res_queue2.qsize() > 0:
            last_value2 = res_queue2.get()
        if not game_module.update_game(last_value1, last_value2):
            break
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

    stream_queue = None  # For Web?
    cp_process = Process(name='game_process', target=game_process, args=(wc_queue, vid_queue, wc_data, video_data, stream_queue))
    cp_process.daemon = True
    cp_process.start()

    while cp_process.is_alive() and vid_process.is_alive() and wc_process.is_alive():
        time.sleep(1)

    print('All Done!')
