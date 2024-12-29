import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
from PIL import Image
import requests

from videoxl.constants import LOGDIR

import pdb

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

try:
    import av
except ImportError:
    print("Please install pyav to use video processing functions.")

def process_video_after_preproecess(video_file):
    # 这类数据已经平均采样过了，所以直接读入图片即可
    frame_names = os.listdir(video_file)
    frame_paths = [ os.path.join(video_file, name) for name in frame_names ]

    # 排序文件名，确保帧的顺序正确（如果文件名是按照帧顺序命名的）
    frame_paths.sort(key=lambda x: int(x.split('.')[-2].split('_')[-1]))

    video_frames = []
    # 读取每一帧图片，转换为RGB格式，并转换为NumPy数组
    for frame_path in frame_paths:
        # 打开图像文件
        with Image.open(frame_path) as img:
            # 转换为RGB格式
            img_rgb = img.convert('RGB')
            # 转换为NumPy数组，并将其添加到帧列表中
            video_frames.append(np.array(img_rgb))

    video = np.stack(video_frames)  # 560x720, 高x宽
    return video

def process_video_with_pyav(video_file, data_args, gt_time_span=None):
    container = av.open(video_file)
    stream = container.streams.video[0]
    avg_fps = round(stream.average_rate / data_args.video_fps)
    
    total_frame_num = stream.frames
    if total_frame_num == 0:
        for frame in container.decode(video=0):
            total_frame_num += 1

    frame_idx_1fps = [i for i in range(0, total_frame_num, avg_fps)] # 如果data_args.video_fps为1，则frame_idx是一秒取一帧
    if data_args.frames_upbound > 0:
        if len(frame_idx_1fps) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        else:
            frame_idx = frame_idx_1fps

    video_frames = []
    video_frames_idx = []
    gt_frame_idx = []

    if gt_time_span is not None:
        candidate_frames = [ [] for _ in gt_time_span ]
        candidate_frames_idx = [ [] for _ in gt_time_span ]

    container.close()
    container = av.open(video_file)
    for index, frame in enumerate(container.decode(video=0)):
        if index in frame_idx:
            if gt_time_span is not None:
                counter_time = index/avg_fps    # 只适用于 data_args.video_fps=1
                for single_gt_time_span in gt_time_span:
                    if single_gt_time_span[0] <= counter_time <= single_gt_time_span[-1]:
                        gt_frame_idx.append( len(video_frames) )
                        break

            video_frames.append(frame.to_rgb().to_ndarray())
            video_frames_idx.append(index)
            if len(video_frames) == len(frame_idx):  # Stop decoding once we have all needed frames
                break

        if index in frame_idx_1fps:
            if gt_time_span is not None:
                counter_time = index/avg_fps   
                for span_idx, single_gt_time_span in enumerate(gt_time_span):
                    if single_gt_time_span[0] <= counter_time <= single_gt_time_span[-1]:
                        candidate_frames[span_idx].append(frame.to_rgb().to_ndarray())
                        candidate_frames_idx[span_idx].append(index)
                        break
        
    container.close()
    if len(gt_frame_idx) == 0 and gt_time_span is not None:  # 采样没采到，需要强行加入
        print(f'触发强行采样机制: {video_file}')
        # 加入到对应位置
        for span_idx, single_gt_time_span in enumerate(gt_time_span):
            # 这个时间范围内第一帧的 idx (1fps original video)
            min_index = candidate_frames_idx[span_idx][0]
            # video_frames_idx 是顺序排列的，从其中找到第一个大于min_index的元素的位置
            # 还有可能根本没有 大于  min_index 的元素
            # 所以默认值改为 len(video_frames_idx)，会被 append 到后面
            insert_index = len(video_frames_idx)
            for pos, index in enumerate(video_frames_idx):
                if index > min_index:
                    insert_index = pos
                    break
            
            video_frames = video_frames[:insert_index] + candidate_frames[span_idx] + video_frames[insert_index:]

            video_frames_idx = video_frames_idx[:insert_index] + candidate_frames_idx[span_idx] + video_frames_idx[insert_index:]

            gt_frame_idx_this_span = list(range(insert_index, insert_index + len(candidate_frames[span_idx])))

            gt_frame_idx.extend(gt_frame_idx_this_span)

    video = np.stack(video_frames)

    return video, gt_frame_idx

# def process_video_with_pyav(video_file, data_args, gt_time_span=None):
#     container = av.open(video_file)
#     stream = container.streams.video[0]
#     avg_fps = round(stream.average_rate / data_args.video_fps)
    
#     total_frame_num = stream.frames

#     all_frames = []
#     total_frame_num=0

#     for frame in container.decode(video=0):
#         total_frame_num += 1
#         all_frames.append(frame.to_rgb().to_ndarray())

#     frame_idx_1fps = [i for i in range(0, total_frame_num, avg_fps)] # 如果data_args.video_fps为1，则frame_idx是一秒取一帧
#     if data_args.frames_upbound > 0:
#         if len(frame_idx_1fps) > data_args.frames_upbound:
#             uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
#             frame_idx = uniform_sampled_frames.tolist()
#         else:
#             frame_idx = frame_idx_1fps

#     video_frames = []
#     video_frames_idx = []
#     gt_frame_idx = []

#     for index, frame in enumerate(all_frames):
#         if index in frame_idx:
#             if gt_time_span is not None:
#                 counter_time = index/avg_fps    # 只适用于 data_args.video_fps=1
#                 # counter_time = frame.pts
#                 for single_gt_time_span in gt_time_span:
#                     if single_gt_time_span[0] <= counter_time <= single_gt_time_span[-1]:
#                         gt_frame_idx.append( len(video_frames) )
#                         break

#             video_frames.append(frame)
#             video_frames_idx.append(index)
#             if len(video_frames) == len(frame_idx):  # Stop decoding once we have all needed frames
#                 break
        
    
#     if len(gt_frame_idx) == 0 and gt_time_span is not None:  # 采样没采到，需要强行加入
#         # print(f'==================检验强行采样机制:==================')
#         # print(f'gt_time_span: {gt_time_span}')
#         # print(f'original video_frames len: {len(video_frames)}')
#         # print(f'original video_frames_idx: {video_frames_idx}')
#         # 先找出候选集
#         candidate_frames = [ [] for _ in gt_time_span ]
#         candidate_frames_idx = [ [] for _ in gt_time_span ]
#         for index in frame_idx_1fps:
#             counter_time = index/avg_fps    # 只适用于 data_args.video_fps=1
#             # counter_time = frame.pts
#             frame = all_frames[index]
#             for span_idx, single_gt_time_span in enumerate(gt_time_span):
#                 if single_gt_time_span[0] <= counter_time <= single_gt_time_span[-1]:
#                     candidate_frames[span_idx].append(frame)
#                     candidate_frames_idx[span_idx].append(index)
#                     break
        
#         # print(f'candidate_frames len: {len(candidate_frames)}')
#         # print(f'candidate_frames_idx: {candidate_frames_idx}')    

#         # random_filter: TODO 随机过滤。 采样没采到这种情况只有视频比较长且clip比较短的时候才会出现，短视频的 frame (1fps) 会被全部加入。暂时全部纳入吧
        
#         # 加入到对应位置
#         for span_idx, single_gt_time_span in enumerate(gt_time_span):
#             # 这个时间范围内第一帧的 idx (1fps original video)
#             min_index = candidate_frames_idx[span_idx][0]
#             # video_frames_idx 是顺序排列的，从其中找到第一个大于min_index的元素的位置
#             # 还有可能根本没有 大于  min_index 的元素
#             # 所以默认值改为 len(video_frames_idx)，会被 append 到后面
#             insert_index = len(video_frames_idx)
#             for pos, index in enumerate(video_frames_idx):
#                 if index > min_index:
#                     insert_index = pos
#                     break
            
#             video_frames = video_frames[:insert_index] + candidate_frames[span_idx] + video_frames[insert_index:]

#             video_frames_idx = video_frames_idx[:insert_index] + candidate_frames_idx[span_idx] + video_frames_idx[insert_index:]

#             gt_frame_idx_this_span = list(range(insert_index, insert_index + len(candidate_frames[span_idx])))

#             gt_frame_idx.extend(gt_frame_idx_this_span)
        
#         # print(f'new video_frames len: {len(video_frames)}')
#         # print(f'new video_frames_idx: {video_frames_idx}')  
#         # print(f'gt_frame_idx: {gt_frame_idx}')

#     else:
#         # print(f'gt_frame_idx: {gt_frame_idx}')
#         # print(f'{video_file} 不需要启动强行采样')
#         pass

#     video = np.stack(video_frames)

#     return video, gt_frame_idx


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
