import torch
from diffusers import AutoPipelineForInpainting
from diffusers import EulerAncestralDiscreteScheduler

from config import model_name, model_dir

inpaintingPipe = AutoPipelineForInpainting.from_pretrained(
  model_name,
  cache_dir = model_dir,
  torch_dtype = torch.float16,
  use_safetensors = True
)

inpaintingPipe.scheduler = EulerAncestralDiscreteScheduler.from_config(inpaintingPipe.scheduler.config)

# inpaintingPipe.to('cuda')
inpaintingPipe.enable_model_cpu_offload()

def inpainting (**props):
  output = inpaintingPipe(**props)
  return output
