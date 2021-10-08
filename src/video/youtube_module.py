from pytube import YouTube
from src.algo.pose_module import VideoPoseDetector
import ffmpeg


# [url, url_start_sec]
all_videos = [
    ['https://www.youtube.com/watch?v=VEWbwlqrYoU&ab_channel=littlesiha', 3],
    ['', -1],
    ['', -1],
    ['', -1]

]

video_dir = "./curr_video"

def progress_handler(progress_info):
    print('{:.2f}'.format(progress_info['percentage']))


def cut_video(file_path, start_sec=0, end_sec=None):
    new_file = file_path + '_new.mp4'
    stream = ffmpeg.input(file_path)
    if end_sec:
        vid = (stream.trim(start=start_sec, end=end_sec).setpts('PTS-STARTPTS'))
        aud = (stream.filter_('atrim', start=start_sec, end=end_sec).filter_('asetpts', 'PTS-STARTPTS'))
    else:
        vid = (stream.trim(start=start_sec).setpts('PTS-STARTPTS'))
        aud = (stream.filter_('atrim', start=start_sec).filter_('asetpts', 'PTS-STARTPTS'))
    joined = ffmpeg.concat(vid, aud, v=1, a=1)
    output = ffmpeg.output(joined, new_file)
    output.run()
    return new_file

def add_pose(file_path):
    vid_detector = VideoPoseDetector(file_path)
    vid_detector.create_pose_data(output='same', visualize=False, print_progress=True)

def replace_chars(text):
    for ch in ["'", "*", "?", ":", "|", "!", "Ø", "&", ","]:
        text = text.replace(ch, "")
    return text

def download_video(url):
    video = YouTube(url)
    res_itag = video.streams.filter(file_extension="mp4", res="360p")[0].itag
    fixed_title = replace_chars(video.title)
    video.streams.get_by_itag(res_itag).download(output_path=video_dir, filename=fixed_title + '.mp4')
    return '{}/{}.mp4'.format(video_dir, fixed_title)


for url, url_start_sec in all_videos:
    if url_start_sec >= 0:
        print('downloading... {}'.format(url))
        file_path = download_video(url)
        print('editing... {}'.format(file_path))
        file_path = cut_video(file_path, start_sec=url_start_sec)
        print('analyzing...{}'.format(file_path))
        add_pose(file_path)
        print('Done!')


