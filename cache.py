import torch
from transparent_background import Remover
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('cache model')

remover = Remover()
model = RealESRGAN(device, scale = 2)
model.load_weights('weights/RealESRGAN_x2.pth', download = True)

print('done')
