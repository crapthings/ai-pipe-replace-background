from PIL import ImageOps
from transparent_background import Remover

remover = Remover()

def remove_background (**props):
  output_image = remover.process(
    props.get('input_image'),
    type = 'map'
  )

  output_image = ImageOps.invert(output_image).convert('RGB')

  return output_image
