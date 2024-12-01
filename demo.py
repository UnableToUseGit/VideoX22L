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

def main():

    model_path = "/share/junjie/shuyan/VideoXL_weight_8"
    video_path="/share/junjie/shuyan/test_demo/ad2_watch_15min.mp4"

    max_frames_num = 900 # you can change this to several thousands so long you GPU memory can handle it :)
    max_frames_num = 256
    gen_kwargs = {"do_sample": True, "temperature": 1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 32}

    attn_implementation='sdpa'
    reload_enable=True
    reload_top_k=3

    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0", attn_implementation=attn_implementation, reload_enable=reload_enable,
    reload_top_k=reload_top_k)

    model.config.beacon_ratio=[8]   # you can delete this line to realize random compression of {2,4,8} ratio

    #video input
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDoes this video contain any inserted advertisement? If yes, which is the content of the ad?<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

    beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
    num_tokens=TOKEN_PERFRAME *max_frames_num
    beacon_skip_last = beacon_skip_first  + num_tokens

    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)

    if IMAGE_TOKEN_INDEX in input_ids:
        transform_input_ids = transform_input_id(input_ids,num_tokens,model.config.vocab_size-1)

    output_ids=output_ids[:,transform_input_ids.shape[1]:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)

if __name__ == '__main__':
    main()
