import os
import torch
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from accelerate import init_empty_weights, infer_auto_device_map
import io
import base64
from PIL import Image

def validate_image_url(image_url, min_resolution=(400, 400)):
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            raise ValueError(f"URL does not point to a valid image (Content-Type: {content_type}).")
        image_data = response.content
        image = Image.open(io.BytesIO(image_data))
        image.verify()
        image = Image.open(io.BytesIO(image_data))
        if image.width < min_resolution[0] or image.height < min_resolution[1]:
            raise ValueError(f"Image resolution ({image.width}x{image.height}) is below the required minimum of {min_resolution[0]}x{min_resolution[1]}.")
        return True
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error or invalid URL. Could not fetch image: {e}")
    except Image.DecompressionBombError:
        raise ValueError("Image is too large and could be a decompression bomb.")
    except Exception as e:
        raise ValueError(f"Image is corrupt or unreadable: {e}")

def validate_and_process_image_input(image_url=None, image_base64=None, min_resolution=(400, 400)):
    if image_base64:
        try:
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            image = Image.open(io.BytesIO(image_data))
            if image.width < min_resolution[0] or image.height < min_resolution[1]:
                raise ValueError(f"Image resolution ({image.width}x{image.height}) is below the minimum of {min_resolution[0]}x{min_resolution[1]}.")
            return image
        except Exception as e:
            raise ValueError(f"Invalid base64 or corrupt image data: {e}")
    elif image_url:
        validate_image_url(image_url, min_resolution)
        return image_url
    else:
        raise ValueError("No image input provided. Please supply either 'image_url' or 'image_base64'.")

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    def process_vision_info(messages):
        images = [
            entry['image']
            for message in messages if message['role'] == 'user'
            for entry in message['content'] if entry['type'] == 'image'
        ]
        return images, None

def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_model_adaptive(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    device = get_available_device()
    if device == "cuda":
        with init_empty_weights():
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")
        device_map = infer_auto_device_map(model, max_memory={0: "7GiB", "cpu": "15GiB"})
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.bfloat16)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    return model, device, 0

def load_processor(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    processor = AutoProcessor.from_pretrained(model_name)
    return processor

def run_inference(model, processor, inputs, max_new_tokens=120):
    messages = inputs['messages']
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        processed_inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **processed_inputs, max_new_tokens=max_new_tokens, temperature=0.6, do_sample=True,
            top_p=0.9, repetition_penalty=1.15, pad_token_id=processor.tokenizer.eos_token_id
        )
        input_token_len = processed_inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, input_token_len:]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return [text.strip() for text in output_text], 0, True
    except Exception as e:
        return None, 0, False