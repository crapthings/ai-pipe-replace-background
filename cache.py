import torch
from transparent_background import Remover
from diffusers import AutoPipelineForInpainting
from RealESRGAN import RealESRGAN

from config import model_name, model_dir

print('cache model')

remover = Remover()

pipe = AutoPipelineForInpainting.from_pretrained(
  model_name,
  cache_dir = model_dir,
  torch_dtype = torch.float16,
  use_safetensors = True
)

model = RealESRGAN('cuda', scale = 2)
model.load_weights('weights/RealESRGAN_x2.pth', download = True)

print('done')
