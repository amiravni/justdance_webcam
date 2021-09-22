### THIS IS BETA!

import time
from flask import Flask, render_template, Response
import cv2
from src.video_module import PlaySyncVideo
from src.main import web_cam_process, compare_poses
from multiprocessing import Process, Queue, current_process

app = Flask(__name__)

#camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
# camera = cv2.VideoCapture(0)
# def gen_frames():  # generate frame by frame from camera
#     while True:
#         # Capture frame-by-frame
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def gen_frames_file(vid_queue):
    file_path = './src/curr_video/Just Dance 2016 - Good Feeling - Flo rida - 5 Stars.mp4'
    return PlaySyncVideo(video_path=file_path, pose_path='same', res_queue=vid_queue, local_video=False)

@app.route('/video_feed')
def video_feed():
    # TODO: make all of this working through the web ( 3 videos + 1 audio)
    #Video streaming route. Put this in the src attribute of an img tag
    wc_queue = Queue()
    wc_process = Process(name='web_cam_process', target=web_cam_process, args=(wc_queue,))
    wc_process.daemon = True
    wc_process.start()
    vid_queue = Queue()

    cp_process = Process(name='compare_poses', target=compare_poses, args=(wc_queue, vid_queue))
    cp_process.daemon = True
    cp_process.start()

    return Response(gen_frames_file(vid_queue), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)