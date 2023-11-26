import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import EulerAncestralDiscreteScheduler

from config import model_dir

img2imgPipe = StableDiffusionImg2ImgPipeline.from_single_file(
  '/runpod-volume/models/realisticVisionV51_v51VAE.safetensors',
  torch_dtype = torch.float16,
  use_safetensors = True
)

img2imgPipe.scheduler = EulerAncestralDiscreteScheduler.from_config(img2imgPipe.scheduler.config)

img2imgPipe.enable_model_cpu_offload()

def img2img (**props):
  output = img2imgPipe(**props)
  return output
