# Task-Aware KV Compression For Cost-Effective Long Video Understanding
        
## Installation
```bash
git clone https://github.com/UnableToUseGit/VideoX22L.git
cd VideoX22L
pip install -r requirements.txt
```

## Download Video-XL Weight
Our approach is training-free based on **Video-XL**. You could download it from [here](https://huggingface.co/sy1998/Video_XL/tree/main).

## Inference
```bash
python inference.py
```

## Reproduce the results of main experiments
We provide the ''.sh'' scripts to reproduce the results of main experiments in ``../lmms-eval/scripts/mainexps/``. You could find scripts with different ''top-k'' in this directory for each benchmark.