import math
import requests

import torch
from diffusers.utils import load_image
import runpod

from utils import extract_origin_pathname, upload_image, rounded_size
from inpainting import inpainting
from img2img import img2img
from remove_background import remove_background
from upscaler import upscale

def run (job, _generator = None):
    # prepare task
    try:
        print('debug', job)

        _input = job.get('input')

        debug = _input.get('debug')
        input_url = _input.get('input_url')
        upload_url = _input.get('upload_url')

        prompt = _input.get('prompt', 'a dog')
        negative_prompt = _input.get('negative_prompt', '')
        num_inference_steps = _input.get('num_inference_steps', 70)
        guidance_scale = _input.get('guidance_scale', 13.0)
        strength = _input.get('strength', 1)
        strength2 = _input.get('strength2')
        upscale = _input.get('upscale')
        seed = _input.get('seed')

        input_image = load_image(input_url).convert('RGB')

        input_copy = input_image.copy()

        limit = 768

        input_copy.thumbnail([limit, limit])

        originalWidth, originalHeight = input_copy.size
        renderWidth, renderHeight = rounded_size(originalWidth, originalHeight)

        mask_image = remove_background(
            input_image = input_copy,
        )

        if seed is not None:
            _generator = torch.Generator(device = 'cuda').manual_seed(seed)

        output_image = inpainting(
            image = input_copy,
            mask_image = mask_image,
            width = renderWidth,
            height = renderHeight,
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps = math.ceil(num_inference_steps / strength),
            guidance_scale = guidance_scale,
            strength = strength,
            generator = _generator
        ).images[0]

        # output_image = output_image.resize(input_image.size)

        if upscale is not None:
            output_image = upscale(output_image)

        if strength2 is not None:
            output_image = img2img(
                image = output_image,
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_inference_steps = math.ceil(num_inference_steps / strength),
                guidance_scale = guidance_scale,
                strength = strength2,
                generator = _generator
            ).images[0]

        if debug:
            output_image.save('sample.png')

        # # output
        output_url = extract_origin_pathname(upload_url)
        output = {
            'input_url': input_url,
            'output_url': output_url
        }

        upload_image(upload_url, output_image)

        return output
    # caught http[s] error
    except requests.exceptions.RequestException as e:
        return { 'error': e.args[0] }

runpod.serverless.start({ 'handler': run })
