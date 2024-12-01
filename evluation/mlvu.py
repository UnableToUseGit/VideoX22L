from videoxl.model.builder import load_pretrained_model
from videoxl.mm_utils import tokenizer_image_token, process_images, transform_input_id
from videoxl.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import json
from tqdm import tqdm
import os
import argparse
from PIL import Image
import random
import numpy as np
from torch.utils.data import Dataset
from videoxl.conversation import conv_templates
from videoxl.constants import IMAGE_TOKEN_INDEX,TOKEN_PERFRAME 
import cv2
import argparse
import pdb
from loguru import logger as eval_logger
import sys
import time

def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        # 移除前导的冒号和空格
        clean_message = message.lstrip(": ").strip()
        if count == len(conv.messages):
            ret += role + " " + clean_message
        else:
            if message:
                ret += role + " " + clean_message + conv.sep
            else:
                ret += role + " "
    return ret


class MLVU(Dataset):
    def __init__(self, data_dir, data_list):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'data': data
                })
        
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': video_path, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }



def check_ans(pred, gt):
    flag = False

    index=gt.index("(")
    index2=gt.index(")")
    gt_option=gt[index+1:index2]

    if ")" in pred:
        index3=pred.index(")")
        pred=pred[index3-1:index3]

    print("2222222",pred,gt_option)
    if pred==gt_option:
        print("11111111111111",pred,gt_option)
        flag=True

    return flag


def parse_args():
    parser = argparse.ArgumentParser(description="Select a task to process")
    parser.add_argument("--tasks", help="The task to process")
    parser.add_argument("--log_level", help="")
    parser.add_argument("--reload_enable", action="store_true", help="") # TODO
    parser.add_argument("--attn_implementation", help="sdpa, flash_attention_2")
    parser.add_argument("--reload_top_k", type=int)
    parser.add_argument("--save_dir", help="")
    
    return parser.parse_args()

args = parse_args()

eval_logger.remove()  # 移除所有默认 handler
log_level = args.log_level
eval_logger.add(sys.stdout, level=log_level)  # 从 INFO 开始输出
eval_logger.debug("This is a DEBUG message")  # 不会显示
eval_logger.info("This is an INFO message")   # 会显示


reload_enable = args.reload_enable

tasks = args.tasks.split(',')
all_data_list = {}
for task in tasks:
    if task=="count":
        data_list = { "count": ("4_count.json",f"/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list", "video")}
    elif task=="ego":
        data_list = {"ego": ("3_ego.json", f"/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list", "video")}
    elif task=="needle":
        data_list = {"needle": ("2_needle.json", f"/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list", "video")}
    elif task=="order":
        data_list = {"order": ("5_order.json", f"/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list", "video")}
    elif task=="plotQA":
        data_list = {"plotQA": ("1_plotQA.json", f"/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list", "video")}
    elif task=="anomaly_reco":
        data_list={"anomaly_reco": ("6_anomaly_reco.json", f"/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list", "video")}
    elif task=="topic_reasoning":
        data_list={"topic_reasoning": ("7_topic_reasoning.json", f"/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list", "video")}

    all_data_list.update(data_list)

data_dir = f"/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/upload_json"

base_folder = "/share/junjie/shuyan/MLVU_dev/frames_256"
dataset = MLVU(data_dir, all_data_list)
# fix seed
torch.manual_seed(0)

model_path = "/share/junjie/shuyan/VideoXL_weight_8"
print("##########",model_path)

 # you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": False, "temperature": 0.0, "top_p": 1, "num_beams": 1, "use_cache": True, "max_new_tokens": 32}
# "sdpa"
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0", attn_implementation=args.attn_implementation, reload_enable=reload_enable,
reload_top_k=args.reload_top_k)

# model.config.beacon_ratio=[8]
# print(f'压缩率：{model.config.beacon_ratio}')

max_frames_num = 256
correct = 0
total = 0
res_list = []
acc_dict = {}

start_time = time.time()

for example in tqdm(dataset):
    model.memory.reset()
    task_type = example['task_type']
    if task_type not in acc_dict:
        acc_dict[task_type] = [0, 0] # correct, total
    acc_dict[task_type][1] += 1
    total += 1
    video_path=example["video"]
    inp=example["question"] + "\nOnly give the best option."

    #video input
    prompt1 = "<|im_start|>system\nCarefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question.<|im_end|>\n<|im_start|>user\n<image>\n"
    # prompt1 = "<|im_start|>\n<image>\n"
    prompt2 = inp
    # prompt2 = ""
    prompt3 = "<|im_end|>\n<|im_start|>assistant\nBest Option: ("
    prompt = prompt1 + prompt2 + prompt3
    # pdb.set_trace()
    print("#####",prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    # vr = VideoReader(video_path, ctx=cpu(0))
    # total_frame_num = len(vr)
    # uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    # frame_idx = uniform_sampled_frames.tolist()
    # frames = vr.get_batch(frame_idx).asnumpy()

    name = video_path.split("/")[-1]
    new_path = os.path.join(base_folder, name)
    num_images = len(os.listdir(new_path))
    print("num_images",num_images)
    frames=[]
    for n in range(num_images):
        image_path = os.path.join(new_path, f"frame_{n:03d}.png")
        with Image.open(image_path) as frame:
            frame = np.array(frame)
            frames.append(frame)

    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

    beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
    num_tokens = TOKEN_PERFRAME * max_frames_num
    beacon_skip_last = beacon_skip_first  + num_tokens

    print(f'beacon_skip_first: {beacon_skip_first}')
    print(f'beacon_skip_last: {beacon_skip_last}')

    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)

    transform_input_ids = transform_input_id(input_ids,num_tokens,model.config.vocab_size-1)

    output_ids = output_ids[:,transform_input_ids.shape[1]:]
    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


    gt = example['answer']
    print("##########")
    print("GT",gt)
    print("Pred",pred)
    print("##########")

    res_list.append({
        'pred': pred,
        'gt': gt,
        'question':example['question'],
        'question_type':example['task_type'],
        'video':example['video']
    })
    if check_ans(pred=pred, gt=gt):
        acc_dict[task_type][0] += 1
        correct += 1
    print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
    print('-' * 30, task_type, '-' * 30)

end_time = time.time()
print(f'{tasks} cousming time: {end_time-start_time}s.')

save_dir = args.save_dir
save_name = '_'.join(tasks) + '_' + 'results.json'
save_path = os.path.join(save_dir, save_name)

with open(save_path, "w") as f:
    json.dump({
        "acc_dict": acc_dict,
        "res_list": res_list
    }, f)

final_res = dict()
total=0
idx=0
for k, v in acc_dict.items():
    idx+=1
    final_res[k] = v[0] / v[1] * 100 
    total+=final_res[k]

final_res['Avg'] = total /idx 
print(final_res)

save_dir = args.save_dir
save_name = '_'.join(tasks) + '_' + 'final_result.json'
save_path = os.path.join(save_dir, save_name)

with open(save_path, "w") as f:
    json.dump(final_res, f)
