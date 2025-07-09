import torch
import requests
import base64
from PIL import Image
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OneThingAIImageToTextNode:
    def __init__(self):
        self.api_key = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "gpt4o", "multiline": False}),
                "retries": ("INT", {"default": 3, "min": 0, "max": 5}),
                "timeout": ("INT", {"default": 20, "min": 5, "max": 100}),
                "max_tokens": ("INT", {"default": 500, "min": 100, "max": 10000, "step": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe_image"
    CATEGORY = "OneThingAI/CV"

    def describe_image(self, image, api_key, model, retries, timeout, max_tokens):
        if not api_key:
            raise ValueError("API key is required")

        # Convert tensor to PIL Image
        if isinstance(image, torch.Tensor):
            # Ensure image is in the correct format (C,H,W) -> (H,W,C)
            image = image.squeeze(0)
            image = (image * 255).byte()
            image = image.cpu().numpy()
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Convert PIL Image to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please describe this image in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            "model": model,
            "max_tokens": max_tokens
        }

        try:
            # Configure retry strategy
            retry_strategy = Retry(
                total=retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            # Create session with retry strategy
            session = requests.Session()
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            
            # Make the request with retry and timeout
            response = session.post(
                "https://api-model.onethingai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the description from the response
            description = result['choices'][0]['message']['content']
            return (description,)
            
        except requests.exceptions.RequestException as e:
            return (f"Error: {str(e)}",)
        finally:
            session.close()

NODE_CLASS_MAPPINGS = {
    "OneThingAI ImageToText": OneThingAIImageToTextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneThingAI ImageToText": "OneThingAI Image Understanding"
} 