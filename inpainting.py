import torch
from diffusers import AutoPipelineForInpainting

from config import model_name, model_dir

inpaintingPipe = AutoPipelineForInpainting.from_pretrained(
  model_name,
  cache_dir = model_dir,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

inpaintingPipe.enable_model_cpu_offload()

def inpainting (**props):
  output = inpaintingPipe(**props)
  return output
