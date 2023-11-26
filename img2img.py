import torch
from diffusers import AutoPipelineForImage2Image

from config import model_name, model_dir

img2imgPipe = AutoPipelineForImage2Image.from_pretrained(
  model_name,
  cache_dir = model_dir,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

img2imgPipe.enable_model_cpu_offload()

def img2img (**props):
  output = img2imgPipe(**props)
  return output
