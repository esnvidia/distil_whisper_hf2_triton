from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch

#from datasets import load_dataset, load_from_disk

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32



model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=False
)

model.save_pretrained('./distil-whisper/distil-large-v2', safe_serialization=False)
print('complete: hf model saved to ./distil-whisper/distil-large-v2')

