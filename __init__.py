import gc
import os
import os.path as osp
import torch
import folder_paths
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
torch_dtype = torch.bfloat16
now_dir = osp.dirname(__file__)
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")

import random
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from huggingface_hub import snapshot_download
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

class LoadQwenImageDiffSynthiPipe:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "offload":("BOOLEAN",{
                    "default":True
                }),
                "fp8_quantization":("BOOLEAN",{
                    "default":False
                }),
            },
            "optional":{
                "lora_model": ("MODEL",),
                "lora_alpha":("FLOAT",{
                    "default":1.0
                })
            }
        }
    
    RETURN_TYPES = ("QwenImageDiffSynthiPipe",)
    RETURN_NAMES = ("pipe",)

    FUNCTION = "load_pipe"

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def load_pipe(self, model, clip, vae, offload, fp8_quantization, lora_model=None, lora_alpha=1.0):
        # Create a custom pipeline using the loaded ComfyUI models
        pipe = QwenImagePipeline()
        
        # Assign the loaded models to the pipeline
        pipe.dit = model.model  # Diffusion transformer
        pipe.text_encoder = clip.cond_stage_model  # Text encoder
        pipe.vae = vae.first_stage_model  # VAE
        
        # Set device and dtype
        pipe.dit.to(device, dtype=torch_dtype)
        pipe.text_encoder.to(device, dtype=torch_dtype)
        pipe.vae.to(device, dtype=torch_dtype)
        
        # Apply quantization if requested
        if fp8_quantization:
            if hasattr(pipe.dit, 'to'):
                pipe.dit = pipe.dit.to(dtype=torch.float8_e4m3fn)
            if hasattr(pipe.text_encoder, 'to'):
                pipe.text_encoder = pipe.text_encoder.to(dtype=torch.float8_e4m3fn)
            if hasattr(pipe.vae, 'to'):
                pipe.vae = pipe.vae.to(dtype=torch.float8_e4m3fn)
        
        # Handle offloading
        if offload:
            # Move models to CPU when not in use
            pipe.enable_model_cpu_offload = True
        
        # Apply LoRA if provided
        if lora_model is not None:
            # Apply LoRA to the diffusion transformer
            pipe.dit = lora_model.model
            # Set LoRA alpha if the model supports it
            if hasattr(pipe.dit, 'set_lora_alpha'):
                pipe.dit.set_lora_alpha(lora_alpha)
        
        # Store original models for potential CPU offloading
        pipe._original_models = {
            'dit': model,
            'clip': clip, 
            'vae': vae,
            'lora': lora_model
        }
        
        return (pipe, )


class LoadQwenImageDiffSynthiPipeControlNet:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "controlnet": ("CONTROL_NET",),
                "offload":("BOOLEAN",{
                    "default":True
                }),
                "fp8_quantization":("BOOLEAN",{
                    "default":False
                }),
            },
            "optional":{
                "lora_model": ("MODEL",),
                "lora_alpha":("FLOAT",{
                    "default":1.0
                })
            }
        }
    
    RETURN_TYPES = ("QwenImageDiffSynthiPipeControlNet",)
    RETURN_NAMES = ("controlnet_pipe",)

    FUNCTION = "load_pipe"

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def load_pipe(self, model, clip, vae, controlnet, offload, fp8_quantization, lora_model=None, lora_alpha=1.0):
        # Create a custom pipeline using the loaded ComfyUI models
        pipe = QwenImagePipeline()
        
        # Assign the loaded models to the pipeline
        pipe.dit = model.model  # Diffusion transformer
        pipe.text_encoder = clip.cond_stage_model  # Text encoder  
        pipe.vae = vae.first_stage_model  # VAE
        pipe.controlnet = controlnet.control_model  # ControlNet
        
        # Set device and dtype
        pipe.dit.to(device, dtype=torch_dtype)
        pipe.text_encoder.to(device, dtype=torch_dtype)
        pipe.vae.to(device, dtype=torch_dtype)
        pipe.controlnet.to(device, dtype=torch_dtype)
        
        # Apply quantization if requested
        if fp8_quantization:
            if hasattr(pipe.dit, 'to'):
                pipe.dit = pipe.dit.to(dtype=torch.float8_e4m3fn)
            if hasattr(pipe.text_encoder, 'to'):
                pipe.text_encoder = pipe.text_encoder.to(dtype=torch.float8_e4m3fn)
            if hasattr(pipe.vae, 'to'):
                pipe.vae = pipe.vae.to(dtype=torch.float8_e4m3fn)
            if hasattr(pipe.controlnet, 'to'):
                pipe.controlnet = pipe.controlnet.to(dtype=torch.float8_e4m3fn)
        
        # Handle offloading
        if offload:
            pipe.enable_model_cpu_offload = True
        
        # Apply LoRA if provided
        if lora_model is not None:
            pipe.dit = lora_model.model
            if hasattr(pipe.dit, 'set_lora_alpha'):
                pipe.dit.set_lora_alpha(lora_alpha)
        
        # Store original models
        pipe._original_models = {
            'dit': model,
            'clip': clip,
            'vae': vae,
            'controlnet': controlnet,
            'lora': lora_model
        }
        
        return (pipe, )


class SetEligenArgs:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "mask":("IMAGE",),
                "prompt":("STRING",),
            },
        }
    RETURN_TYPES = ("EligenArgs","IMAGE",)
    RETURN_NAMES = ("eligen_args","mask",)

    FUNCTION = "set_args"

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def set_args(self,mask,prompt):
        eligen_args = dict(masks=[comfy2pil(mask)],prompts=[prompt])
        mask = visualize_masks(comfy2pil(mask),masks=eligen_args['masks'],
                               mask_prompts=eligen_args['prompts'])
        return (eligen_args,pil2comfy(mask),)


class EligenArgsConcat:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "a_eligen_args":("EligenArgs",),
                "b_eligen_args":("EligenArgs",),
            },
            "optional":{
                "c_eligen_args":("EligenArgs",),
            }
        }
    RETURN_TYPES = ("EligenArgs","IMAGE",)
    RETURN_NAMES = ("eligen_args","mask",)

    FUNCTION = "set_args"

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def set_args(self,a_eligen_args,b_eligen_args,c_eligen_args=None):
        masks=a_eligen_args["masks"]+b_eligen_args["masks"]
        prompts=a_eligen_args["prompts"]+b_eligen_args["prompts"]
        if c_eligen_args is not None:
            masks += c_eligen_args["masks"]
            prompts += c_eligen_args["prompts"]
        eligen_args = dict(masks=masks, prompts=prompts)
        
        empty_pil = Image.new("RGB",size=eligen_args['masks'][0].size,color=0)
        mask = visualize_masks(empty_pil,masks=masks,mask_prompts=prompts)
        
        return (eligen_args,pil2comfy(mask),)


class QwenImageRatio2Size:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "aspect_ratio":(["1:1","16:9","9:16","4:3","3:4"],)
            }
        }
    
    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width","height",)

    FUNCTION = "get_image_size"

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def get_image_size(self,aspect_ratio):
        if aspect_ratio == "1:1":
            return (1328, 1328,)
        elif aspect_ratio == "16:9":
            return (1664, 928,)
        elif aspect_ratio == "9:16":
            return (928, 1664,)
        elif aspect_ratio == "4:3":
            return (1472, 1140,)
        elif aspect_ratio == "3:4":
            return (1140, 1472,)
        else:
            return (1328, 1328,)
        

class QwenImageDiffSynthSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "pipe":("QwenImageDiffSynthiPipe",),
                "prompt":("STRING",),
                "negative_prompt":("STRING",),
                "width":("INT",{
                    "default":982
                }),
                "height":("INT",{
                    "default":1664
                }),
                "num_inference_steps":("INT",{
                    "default":30
                }),
                "guidance_scale":("FLOAT",{
                    "default":4,
                }),
                "seed":("INT",{
                    "default":42
                }),
            },
            "optional":{
                "eligen_args":("EligenArgs",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "sample"

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def sample(self,pipe,prompt,negative_prompt,
               width,height,num_inference_steps,
               guidance_scale,seed,eligen_args=None):
        if eligen_args is None:
            eligen_args = dict(masks=None,prompts=None)
        masks = eligen_args["masks"]
        prompts = eligen_args["prompts"]
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        image = pipe(
            prompt=prompt,
            cfg_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            eligen_entity_prompts=prompts,
            eligen_entity_masks=masks,
        )
        
        return (pil2comfy(image),)


class QwenImageControlNetSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "controlnet_pipe":("QwenImageDiffSynthiPipeControlNet",),
                "prompt":("STRING",),
                "controlnet_image":("IMAGE",),
                "width":("INT",{
                    "default":1328
                }),
                "height":("INT",{
                    "default":1328
                }),
                "num_inference_steps":("INT",{
                    "default":30
                }),
                "guidance_scale":("FLOAT",{
                    "default":4,
                }),
                "seed":("INT",{
                    "default":42
                }),
            },
            "optional":{
                "negative_prompt":("STRING",{
                    "default":""
                }),
                "controlnet_scale":("FLOAT",{
                    "default":1.0
                }),
                "eligen_args":("EligenArgs",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "sample"

    CATEGORY = "AIFSH/QwenImageDiffSynth"

    def sample(self,controlnet_pipe,prompt,controlnet_image,
               width,height,num_inference_steps,
               guidance_scale,seed,negative_prompt="",
               controlnet_scale=1.0,eligen_args=None):
        from diffsynth.pipelines.qwen_image import ControlNetInput
        
        # Convert ComfyUI image to PIL
        controlnet_pil = comfy2pil(controlnet_image).resize((width, height))
        
        # Prepare ControlNet input
        controlnet_input = ControlNetInput(image=controlnet_pil, scale=controlnet_scale)
        
        # Prepare EliGen args if provided
        kwargs = {}
        if eligen_args is not None:
            masks = eligen_args.get("masks")
            prompts = eligen_args.get("prompts")
            if masks and prompts:
                kwargs["eligen_entity_prompts"] = prompts
                kwargs["eligen_entity_masks"] = masks
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate image
        image = controlnet_pipe(
            prompt=prompt,
            cfg_scale=guidance_scale,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            blockwise_controlnet_inputs=[controlnet_input],
            **kwargs
        )
        
        return (pil2comfy(image),)
    

def comfy2pil(image):
    i = 255. * image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img
    
def pil2comfy(pil):
    image = np.array(pil).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

def visualize_masks(image, masks, mask_prompts,font_size=35, use_random_colors=False):
    # Create a blank image for overlays
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

    colors = [
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
    ]
    # Generate random colors for each mask
    if use_random_colors:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 80) for _ in range(len(masks))]

    # Font settings
    try:
        font = ImageFont.truetype(osp.join(now_dir,"font/Arial-Unicode-Regular.ttf"), font_size)  # Adjust as needed
    except IOError:
        font = ImageFont.load_default(font_size)

    # Overlay each mask onto the overlay image
    for mask, mask_prompt, color in zip(masks, mask_prompts, colors):
        # Convert mask to RGBA mode
        mask_rgba = mask.convert('RGBA')
        mask_data = mask_rgba.getdata()
        new_data = [(color if item[:3] == (255, 255, 255) else (0, 0, 0, 0)) for item in mask_data]
        mask_rgba.putdata(new_data)

        # Draw the mask prompt text on the mask
        draw = ImageDraw.Draw(mask_rgba)
        mask_bbox = mask.getbbox()  # Get the bounding box of the mask
        if mask_bbox:  # Check if mask is not empty
            text_position = (mask_bbox[0] + 10, mask_bbox[1] + 10)  # Adjust text position based on mask position
            draw.text(text_position, mask_prompt, fill=(255, 255, 255, 255), font=font)

        # Alpha composite the overlay with this mask
        overlay = Image.alpha_composite(overlay, mask_rgba)

    # Composite the overlay onto the original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)

    return result


NODE_CLASS_MAPPINGS = {
    "LoadQwenImageDiffSynthiPipe": LoadQwenImageDiffSynthiPipe,
    "LoadQwenImageDiffSynthiPipeControlNet": LoadQwenImageDiffSynthiPipeControlNet,
    "QwenImageDiffSynthSampler":QwenImageDiffSynthSampler,
    "QwenImageControlNetSampler":QwenImageControlNetSampler,
    "QwenImageRatio2Size":QwenImageRatio2Size,
    "SetEligenArgs":SetEligenArgs,
    "EligenArgsConcat":EligenArgsConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwenImageDiffSynthiPipe": "LoadQwenImageDiffSynthiPipe@关注超级面爸微信公众号",
    "LoadQwenImageDiffSynthiPipeControlNet": "LoadQwenImageDiffSynthiPipeControlNet@关注超级面爸微信公众号",
    "QwenImageDiffSynthSampler":"QwenImageDiffSynthSampler@关注超级面爸微信公众号",
    "QwenImageControlNetSampler":"QwenImageControlNetSampler@关注超级面爸微信公众号",
    "QwenImageRatio2Size":"QwenImageRatio2Size@关注超级面爸微信公众号",
    "SetEligenArgs":"SetEligenArgs@关注超级面爸微信公众号",
    "EligenArgsConcat":"EligenArgsConcat@关注超级面爸微信公众号",
}
