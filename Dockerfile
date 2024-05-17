FROM nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3

RUN pip install tiktoken kaldialign openai-whisper tritonclient


