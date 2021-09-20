from pytube import YouTube
from pose_module import VideoPoseDetector

def add_pose(video_path):
    pass

video = YouTube('https://www.youtube.com/watch?v=VDR-jJnVqPc')
res_itag = video.streams.filter(file_extension="mp4", res="720p")[0].itag
video.streams.get_by_itag(res_itag).download("./curr_video")

