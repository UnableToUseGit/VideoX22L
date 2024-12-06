from videoxl.model.builder import load_pretrained_model
from videoxl.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videoxl.constants import IMAGE_TOKEN_INDEX,TOKEN_PERFRAME 
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import pdb
from loguru import logger as eval_logger
import sys
import cProfile

eval_logger.remove()  # 移除所有默认 handler
eval_logger.add(sys.stdout, level="INFO")  # 从 INFO 开始输出
eval_logger.debug("This is a DEBUG message")  # 不会显示
eval_logger.info("This is an INFO message")   # 会显示

# fix seed
torch.manual_seed(0)


cases = {
    'needle_case_1':{
        'video': "/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list/needle_99.mp4",
        "question": "What time of day is it when the young girl in a tracksuit is doing yoga in the park?",
        "candidates": ["Morning", "Midday", "Sunset", "Night"],
        "video_start_sec": 198.6683943197922,
        "video_end_sec": 210.3683943197922
    },
    'needle_case_2':{
        'video': "/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list/needle_40.mp4",
        "question": "What is the nature of the den where the American toad is sitting in the video?", 
        "candidates": ["Stone cavity", "Earthen cavity", "Wooden cavity", "Water cavity"],
        "video_start_sec": 241.00993495187757,
        "video_end_sec": 268.0799349518776
    }
}

def main():

    case_name = 'needle_case_1'
    case = cases[case_name]

    model_path = "/share/shuyan/VideoXL_weight_8"
    video_path = case['video']
    # video_path="/share/junjie/code/videofactory/Evaluation_LVBench/LVBench_all/video_list/needle_40.mp4"

    max_frames_num = 256
    gen_kwargs = {"do_sample": False, "temperature": 1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 32}

    attn_implementation='sdpa'
    reload_enable=True
    reload_top_k=30

    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0", attn_implementation=attn_implementation, reload_enable=reload_enable,
    reload_top_k=reload_top_k)

    # TODO change setting
    model.config.beacon_ratio=[8]   # you can delete this line to realize random compression of {2,4,8} ratio

    while True:
        question = f"Question: {case['question']}\n"
        question += "Options:\n"
        for idx, c in enumerate(case['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
        question = question.rstrip()

        inp = question + "\nOnly give the best option."

        #video input
        prompt1 = "<|im_start|>system\nCarefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question.<|im_end|>\n<|im_start|>user\n<image>\n"
        prompt2 = inp
        prompt3 = "<|im_end|>\n<|im_start|>assistant\nBest Option: ("
        prompt = prompt1 + prompt2 + prompt3

        # change setting
        # query_suffix = '\n' + prompt2 + prompt3
        query_suffix = "young girl in a tracksuit is doing yoga"
        print(f'query_suffix: {query_suffix}')

        model.memory.query_suffix = query_suffix
        query_suffix_token_ids = tokenizer_image_token(query_suffix, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)  
        model.memory.query_suffix_token_ids = query_suffix_token_ids
        model.memory.query_suffix_token_ids_length = query_suffix_token_ids.shape[-1]

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

        beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
        num_tokens = TOKEN_PERFRAME *max_frames_num
        beacon_skip_last = beacon_skip_first  + num_tokens

        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)

        if IMAGE_TOKEN_INDEX in input_ids:
            transform_input_ids = transform_input_id(input_ids,num_tokens,model.config.vocab_size-1)

        output_ids=output_ids[:,transform_input_ids.shape[1]:]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)

        model.memory.reset()

if __name__ == '__main__':
    main()
