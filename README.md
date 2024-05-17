# distil_whisper_hf2_triton

# Steps:

```bash
git clone https://github.com/esnvidia/distil_whisper_hf2_triton.git

cd  distil_whisper_hf2_triton
git submodule update --init --recursive

docker build -t distil_whisper_hf2_triton -f Dockerfile .

docker run --name distil_whisper_triton --rm -it --gpus all --net host  -v `pwd`:/workspace distil_whisper_hf2_triton bash
```

Inside the container:

```bash
cd /workspace
```

# save the HF model
```bash
python save_distil_whisper_from_hf.py
```
# Following many of the same steps [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper#distil-whisper):

# Convert the model from .bin to .pt
```
python TensorRT-LLM/examples/whisper/distil_whisper/convert_from_distil_whisper.py --model_name ./distil-whisper/distil-large-v2 --cache_dir ./distil-whisper/distil-large-v2/ --output_dir ./assets/ --output_name distil-large-v2
```

# Build the engine

``` 
output_dir=distil_whisper_large_v2
python TensorRT-LLM/examples/whisper/build.py --model_name distil-large-v2 --output_dir $output_dir --dtype float16 --enable_context_fmha --use_gpt_attention_plugin --use_gemm_plugin --use_bert_attention_plugin float16
```
# Test (Long running, feel free to skip)
```
python run.py --engine_dir $output_diry --name librispeech_dummy_output --tokenizer_name gpt2 --assets_dir ./assets/ --dataset librispeech_asr --results_dir ./results
```

#create model repo with Python Backend (sherpa)
```
cp -r distil_whisper_large_v2/ model_repo_whisper_trtllm/whisper/1/
cp -r model_repo_whisper_trtllm ./sherpa/triton/whisper/
```

# Run Triton Server
```bash
cd /workspace/sherpa/triton/whisper

tritonserver --model-repository model_repo_whisper_trtllm/

```

# From another terminal

```
docker exec -it distil_whisper_triton bash
cd /workspace/Triton-ASR-Client
num_task=16
python client.py --server-addr localhost --model-name whisper --num-tasks $num_task --whisper-prompt "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" --manifest-dir ./datasets/mini_en/
```


Thanks to the authors of the TRT-LLM, sherpa and Triton-ASR-Client repos for their contribution. Heavily used their contributions to show how a distil whisper model from huggingface can be converted to a `.pt` format, then optimized with TRT-LLM, loaded/served into Triton, and send requests.


WER needs improvement. Filed issue w/ TRT-LLM team.
