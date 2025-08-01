import os
import time
import psutil
import torch
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from accelerate import init_empty_weights, infer_auto_device_map
import io
import base64
from PIL import Image

def validate_image_url(image_url, min_resolution=(400, 400)):
    """
    Validates an image URL before processing (kept for backward compatibility).
    """
    print(f"Validating image URL: {image_url}...")
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
            
        print("Image validation successful.")
        return True

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error or invalid URL. Could not fetch image: {e}")
    except Image.DecompressionBombError:
        raise ValueError("Image is too large and could be a decompression bomb.")
    except Exception as e:
        raise ValueError(f"Image is corrupt or unreadable: {e}")

def validate_and_process_image_input(image_url=None, image_base64=None, min_resolution=(400, 400)):
    """
    Validates image input from a URL or base64 string and returns a processable object.
    
    Returns:
        (str or PIL.Image.Image): The validated URL string or the loaded PIL Image object.
    Raises:
        ValueError: If input is invalid or not provided.
    """
    if image_base64:
        print("Validating base64 image input...")
        try:
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image.verify()

            image = Image.open(io.BytesIO(image_data))
            if image.width < min_resolution[0] or image.height < min_resolution[1]:
                raise ValueError(f"Image resolution ({image.width}x{image.height}) is below the minimum of {min_resolution[0]}x{min_resolution[1]}.")

            print("Base64 image validation successful.")
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
    print("Warning: qwen_vl_utils.py not found. Please ensure it is in the same directory.")
    def process_vision_info(messages):
        images = [
            entry['image']
            for message in messages if message['role'] == 'user'
            for entry in message['content'] if entry['type'] == 'image'
        ]
        return images, None

def setup_environment():
    """Sets an environment variable for better PyTorch CUDA memory management."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Environment configured for potentially better memory management.")

def get_available_device():
    """
    Checks for available hardware (GPU or CPU) and returns the appropriate device string.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {device_name} ({memory_gb:.1f}GB VRAM)")
        return "cuda"
    else:
        cpu_count = psutil.cpu_count(logical=True)
        ram_gb = psutil.virtual_memory().total / 1024**3
        print(f"No GPU detected. Using CPU ({cpu_count} cores, {ram_gb:.1f}GB RAM).")
        return "cpu"

def load_model_adaptive(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    """
    Loads the Qwen2-VL model adaptively based on the available hardware.
    """
    device = get_available_device()
    print(f"Loading model '{model_name}' onto {device.upper()}...")
    start_time = time.time()
    
    if device == "cuda":
        with init_empty_weights():
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")
        device_map = infer_auto_device_map(model, max_memory={0: "7GiB", "cpu": "15GiB"})
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.bfloat16)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f} seconds.")
    return model, device, load_time

def load_processor(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    """
    Loads the processor associated with the specified model.
    """
    print(f"Loading processor for '{model_name}'...")
    processor = AutoProcessor.from_pretrained(model_name)
    print("Processor loaded.")
    return processor

def run_inference(model, processor, inputs, max_new_tokens=120):
    """
    Runs inference using the model to generate text based on the provided inputs.
    """
    print("Starting inference process...")
    start_time = time.time()
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
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds.")
        return [text.strip() for text in output_text], inference_time, True
        
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"An error occurred during inference: {e}")
        return None, inference_time, False