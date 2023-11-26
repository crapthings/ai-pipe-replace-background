import torch
from diffusers import AutoPipelineForText2Image
from config import model_name, model_dir

print('cache model')

pipe = AutoPipelineForText2Image.from_pretrained(
  model_name,
  cache_dir = model_dir,
  torch_dtype = torch.float16,
  variant = 'fp16',
  use_safetensors = True
)

print('done')
