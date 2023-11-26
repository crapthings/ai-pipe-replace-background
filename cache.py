import torch
from transparent_background import Remover
from RealESRGAN import RealESRGAN

print('cache model')

remover = Remover()
model = RealESRGAN('cuda', scale = 2)
model.load_weights('weights/RealESRGAN_x2.pth', download = True)

print('done')
