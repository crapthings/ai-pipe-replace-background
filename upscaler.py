from RealESRGAN import RealESRGAN

model = RealESRGAN('cuda', scale = 2)
model.load_weights('weights/RealESRGAN_x2.pth', download = True)

def upscale (input_image):
    output_image = model.predict(input_image)
    return output_image
