import os
import torch
import folder_paths
import comfy.utils
import comfy.model_management
import numpy as np
from PIL import Image
import cv2

class QwenImageControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"controlnet_name": (folder_paths.get_filename_list("controlnet"), )}}
    
    RETURN_TYPES = ("CONTROLNET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders/qwen_image"

    def load_controlnet(self, controlnet_name):
        controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
        controlnet_data = comfy.utils.load_torch_file(controlnet_path)
        
        # Check if this is a valid Qwen-Image ControlNet
        if not self.is_qwen_image_controlnet(controlnet_data):
            raise RuntimeError("ERROR: Selected model is not a valid Qwen-Image ControlNet")
        
        # Process state dict to remove prefix if needed
        prefix_to_remove = "pipe.blockwise_controlnet.models.0."
        processed_data = {}
        for k, v in controlnet_data.items():
            if k.startswith(prefix_to_remove):
                processed_data[k[len(prefix_to_remove):]] = v
            else:
                processed_data[k] = v
                
        return ({"name": controlnet_name, "controlnet_data": processed_data},)

    def is_qwen_image_controlnet(self, state_dict):
        # Check for blockwise ControlNet specific layers
        has_add_projections = any("add_q_proj" in k for k in state_dict.keys())
        has_blockwise_prefix = any("blockwise_controlnet" in k for k in state_dict.keys())
        
        # Should NOT have standard ControlNet layers
        has_standard_controlnet = any("controlnet_cond_embedding" in k for k in state_dict.keys())
        
        return has_add_projections and (has_blockwise_prefix or not has_standard_controlnet)


class QwenImageControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                            "controlnet": ("CONTROLNET",),
                            "image": ("IMAGE",),
                            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})}}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "model/controlnet"

    def apply_controlnet(self, model, controlnet, image, strength):
        # Clone the model to avoid modifying the original
        m = model.clone()
        
        # Process the image to get conditioning
        conditioning = self.process_conditioning(image)
        
        # Store the conditioning for later use
        m.model_options.setdefault("transformer_options", {})
        m.model_options["transformer_options"]["blockwise_controlnet"] = {
            "conditioning": conditioning,
            "strength": strength,
            "controlnet_data": controlnet["controlnet_data"]
        }
        
        # Patch the attention layers
        self.patch_attention_layers(m.model.diffusion_model)
        
        return (m,)
    
    def process_conditioning(self, image):
        # Convert from ComfyUI format [B, H, W, C] to [B, C, H, W]
        image = image.permute(0, 3, 1, 2)
        # Scale to -1 to 1 range
        conditioning = 2.0 * image - 1.0
        return conditioning
    
    def patch_attention_layers(self, transformer):
        # Iterate through all transformer blocks
        for i, block in enumerate(transformer.transformer_blocks):
            # Check if the attention layer has the required projection layers
            if hasattr(block.attn, "add_q_proj") and hasattr(block.attn, "add_k_proj") and hasattr(block.attn, "add_v_proj"):
                # Store the original forward method
                if not hasattr(block.attn, "original_forward"):
                    block.attn.original_forward = block.attn.forward
                    
                    # Create a patched forward method
                    def make_patched_forward(attn_layer, block_idx):
                        def patched_forward(hidden_states, encoder_hidden_states=None, 
                                          encoder_hidden_states_mask=None, image_rotary_emb=None, **kwargs):
                            # Get ControlNet data
                            transformer_options = attn_layer.model_options.get("transformer_options", {})
                            controlnet = transformer_options.get("blockwise_controlnet", None)
                            
                            # Call original forward
                            img_attn_output, txt_attn_output = attn_layer.original_forward(
                                hidden_states, encoder_hidden_states, 
                                encoder_hidden_states_mask, image_rotary_emb, **kwargs
                            )
                            
                            if controlnet is not None and encoder_hidden_states is not None:
                                # Apply ControlNet conditioning
                                conditioning = controlnet["conditioning"]
                                strength = controlnet["strength"]
                                controlnet_data = controlnet["controlnet_data"]
                                
                                # In Qwen-Image, the ControlNet is applied through the add_*_proj layers
                                # These layers process the additional conditioning information
                                
                                # Get the ControlNet weights for this block
                                q_proj_key = f"transformer_blocks.{block_idx}.attn.add_q_proj.weight"
                                k_proj_key = f"transformer_blocks.{block_idx}.attn.add_k_proj.weight"
                                v_proj_key = f"transformer_blocks.{block_idx}.attn.add_v_proj.weight"
                                
                                # Apply ControlNet conditioning if weights exist
                                if (q_proj_key in controlnet_data and 
                                    k_proj_key in controlnet_data and 
                                    v_proj_key in controlnet_data):
                                    
                                    # Process the conditioning through the ControlNet layers
                                    # This is where the actual ControlNet application happens
                                    batch_size = conditioning.shape[0]
                                    
                                    # Reshape conditioning to match expected dimensions
                                    cond_reshaped = conditioning.view(batch_size, conditioning.shape[1], -1).permute(0, 2, 1)
                                    
                                    # Apply ControlNet weights
                                    q_cond = torch.nn.functional.linear(
                                        cond_reshaped, 
                                        controlnet_data[q_proj_key]
                                    )
                                    k_cond = torch.nn.functional.linear(
                                        cond_reshaped, 
                                        controlnet_data[k_proj_key]
                                    )
                                    v_cond = torch.nn.functional.linear(
                                        cond_reshaped, 
                                        controlnet_data[v_proj_key]
                                    )
                                    
                                    # Apply strength scaling
                                    q_cond = q_cond * strength
                                    k_cond = k_cond * strength
                                    v_cond = v_cond * strength
                                    
                                    # Add to the attention outputs
                                    img_attn_output = img_attn_output + q_cond
                                    txt_attn_output = txt_attn_output + k_cond
                            
                            return img_attn_output, txt_attn_output
                        
                        return patched_forward
                    
                    # Apply the patched forward method
                    block.attn.forward = make_patched_forward(block.attn, i)
        
        return transformer


class QwenImageControlNetPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                            "control_type": (["canny", "depth", "hed", "scribble", "none"], {"default": "canny"})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"
    CATEGORY = "preprocessors/qwen_image"

    def preprocess(self, image, control_type):
        # Convert from [B, H, W, C] to [H, W, C] for processing (take first image if batch)
        image_np = 255. * image[0].cpu().numpy()
        
        # Apply appropriate preprocessing based on control type
        if control_type == "canny":
            # Apply Canny edge detection
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            result = edges / 255.0
            # Convert back to 3 channels
            result = np.stack([result, result, result], axis=-1)
        elif control_type == "depth":
            # Apply depth estimation (would require a depth model)
            # This is simplified - in reality you'd use a depth estimation model
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            result = gray / 255.0
            # Convert back to 3 channels
            result = np.stack([result, result, result], axis=-1)
        elif control_type == "hed":
            # HED edge detection (simplified)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            result = edges / 255.0
            result = np.stack([result, result, result], axis=-1)
        elif control_type == "scribble":
            # Convert to grayscale and invert (for scribble-like effect)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            result = 1.0 - (gray / 255.0)
            result = np.stack([result, result, result], axis=-1)
        else:  # "none"
            # Just return the original image normalized to 0-1
            result = image_np / 255.0
        
        # Convert back to tensor format ComfyUI expects [1, H, W, C]
        result_tensor = torch.from_numpy(result).unsqueeze(0).float()
        
        return (result_tensor,)


# Register the nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenImageControlNetLoader": QwenImageControlNetLoader,
    "QwenImageControlNetApply": QwenImageControlNetApply,
    "QwenImageControlNetPreprocessor": QwenImageControlNetPreprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageControlNetLoader": "Qwen Image ControlNet Loader",
    "QwenImageControlNetApply": "Qwen Image ControlNet Apply",
    "QwenImageControlNetPreprocessor": "Qwen Image ControlNet Preprocessor"
}
