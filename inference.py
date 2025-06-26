import torch
import sys
sys.path.append("./videox22l")
from videox22l.model.builder import load_pretrained_model
from videox22l.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videox22l.constants import IMAGE_TOKEN_INDEX,TOKEN_PERFRAME 
from decord import VideoReader, cpu
import numpy as np
import pdb

# load model:
model_path = '/share/junjie/shuyan/VideoXL_weight_8'
model_name = 'llava_qwen'
device_map = 'cuda:0'
L_KV_compression_ratio = 2
H_KV_compression_ratio = 32
llava_model_args = {}
llava_model_args["overwrite_config"] = {'high_cmpr_ratio':H_KV_compression_ratio, 'low_cmpr_ratio':L_KV_compression_ratio}
tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, model_name, device_map=device_map, attn_implementation='sdpa', **llava_model_args)

# prepare input
video_path = '/share/minghao/qiM8r7Ft9Lc_44.mp4'
max_frames_num = 100
chunk_size = 10 # default=10, It means there are 10 chunks in total, 10 frame per chunk.
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
vr = VideoReader(video_path, ctx=cpu(0))
total_frame_num = len(vr)
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
frame_idx = uniform_sampled_frames.tolist()
frames = vr.get_batch(frame_idx).asnumpy()
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
num_tokens = TOKEN_PERFRAME * max_frames_num
beacon_skip_last = beacon_skip_first  + num_tokens

print("\n" + "="*40)
print("✨ Frame Processing Configuration ✨")
print("="*40)
print(f"  Chunk Size (frames per segment)   : {chunk_size}")
print(f"  Total Frames                      : {max_frames_num}")
print(f"  Calculated Number of Chunks       : {round(max_frames_num//chunk_size)}")
print(f"  Tokens per Chunk (Window Context) : {TOKEN_PERFRAME*chunk_size} tokens")
print("="*40 + "\n")

# Pre-filling system prompt and visual input to generate bi-level KVs.
with torch.inference_mode():
    system_kvs, video_bi_kvs = model.prefill_bi_kvs(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, use_cache=True, chunk_size=chunk_size)

"""
the structure of video_bi_kvs
cmpr_ratio: bi-level (H_KV_compression_ratio and L_KV_compression_ratio)
layer_idx: 0-28
chunk_idx: decided by frame numbers and chunk size (default=10)
video_k = video_bi_kvs[cmpr_ratio][layer_idx][chunk_idx][0]
video_v = video_bi_kvs[cmpr_ratio][layer_idx][chunk_idx][1]
sys_k = system_kvs[layer_idx][1]
sys_v = system_kvs[layer_idx][1]
"""

# offload these KVs to cpu memory
def offload_to_cpu(data_structure):
    """
    Recursively moves all torch.Tensor objects within a nested
    list, tuple, or dictionary to CPU.
    """
    if isinstance(data_structure, torch.Tensor):
        return data_structure.cpu()
    elif isinstance(data_structure, (list, tuple)):
        # If it's a list or tuple, iterate and apply recursively
        return type(data_structure)([offload_to_cpu(item) for item in data_structure])
    elif isinstance(data_structure, dict):
        # If it's a dictionary, iterate through values and apply recursively
        return {key: offload_to_cpu(value) for key, value in data_structure.items()}
    else:
        # If it's not a Tensor, list, tuple, or dict, return as is
        return data_structure

system_kvs_offloaded = offload_to_cpu(system_kvs)
video_bi_kvs_offloaded = offload_to_cpu(video_bi_kvs)

# Downstream Processing:
# Leveraging the pre-computed and offloaded Key-Value (KV) caches enables flexible and efficient downstream processing.
# This design offers significant practical advantages for various applications.

# 1. Selective KV Reloading based on Frame Importance:
#    This step involves reloading only the KV caches corresponding to "important" frame chunks.
#    The determination of important frames can be customized based on specific task requirements.
#    The most simple way to use an MLLM (Multi-Modal Large Language Model) Retriever
#    to identify the most relevant video chunks.
selected_chunk_idx_mock = [3,5,7] # Example: Indices of selected important chunks (e.g., 3rd, 5th, and 7th out of total 10 chunks).

# 2. Bi-Level Decoding:
#    With the extensive visual KVs pre-filled and available, the system can efficiently handle
#    numerous subsequent user queries or multi-turn interactions without re-processing
#    the entire video input, significantly improving inference speed and interactivity.

suffix = '\n<|im_end|>\n<|im_start|>assistant\n'

user_input_list = [
    "Please describe this video.",
    "What happened in this video?",
    "What did the author say about this video?",
]

gen_kwargs = {"do_sample": False, "temperature": 0.2, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 64}

# --- Optional: For colored output ---
try:
    from colorama import Fore, Style, init
    init(autoreset=True) # Automatically reset color after each print
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False
    print("Colorama not installed. Output will not be colored.")
# ------------------------------------

print(f"\n{'='*15} STARTING MODEL INFERENCE {'='*15}\n")

for i, user_input in enumerate(user_input_list):
    prompt = f"\n{user_input}" + suffix
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    model.is_the_query_input_ids = True
    with torch.inference_mode():
        output_ids = model.bi_level_decoding(input_ids, system_kvs_offloaded=system_kvs_offloaded, video_bi_kvs_offloaded=video_bi_kvs_offloaded, selected_chunk_idx=selected_chunk_idx_mock, **gen_kwargs)
        model.memory.reset()

    output_ids=output_ids[:,input_ids.shape[1]:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # --- Enhanced Print Output ---
    if COLOR_ENABLED:
        print(f"{Fore.BLUE}{Style.BRIGHT}--- QUERY {i+1} ---{Style.RESET_ALL}")
        print(f"{Fore.CYAN}User Input:{Style.RESET_ALL} {user_input}")
        # Optionally, print the full prompt_for_model if useful for debugging
        # print(f"{Fore.YELLOW}Full Prompt (for model):{Style.RESET_ALL} {prompt_for_model.replace('<|im_end|>', '<END>').replace('<|im_start|>', '<START>')}")
        print(f"{Fore.GREEN}Model Output:{Style.RESET_ALL} {outputs}")
    else:
        print(f"--- QUERY {i+1} ---")
        print(f"User Input: {user_input}")
        # print(f"Full Prompt (for model): {prompt_for_model.replace('<|im_end|>', '<END>').replace('<|im_start|>', '<START>')}")
        print(f"Model Output: {outputs}")

    # Add a clear separator for the next query or the end
    if i < len(user_input_list) - 1:
        print("\n" + "="*50 + "\n") # Separator between queries
    else:
        print(f"\n{'='*15} INFERENCE COMPLETE {'='*15}\n") # End message
