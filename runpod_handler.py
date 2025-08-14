import runpod
import json
from model_utils import setup_environment, load_model_adaptive, load_processor, run_inference, validate_and_process_image_input
from prompt_engineering import create_analysis_prompt, create_placement_prompt

model, processor, config = None, None, None

def load_essentials():
    global model, processor, config
    
    if model is None or processor is None:
        setup_environment()
        model, _, _ = load_model_adaptive()
        processor = load_processor()

    if config is None:
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise RuntimeError("FATAL: config.json not found.")

def handler(job):
    job_input = job.get('input', {})
    load_essentials()

    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")
    
    try:
        image_input = validate_and_process_image_input(image_url=image_url, image_base64=image_base64)
    except ValueError as e:
        return {"error": f"Image Input Error: {e}"}

    room_type = job_input.get("room_type", "living room")
    style = job_input.get("style", "industrial")
    max_tokens = job_input.get("max_tokens", 180)
    important_prompt = job_input.get("important_prompt", "")

    furniture_config = config.get("FURNITURE_CONFIG", {})
    style_materials = config.get("STYLE_MATERIALS", {})

    _, analysis_messages = create_analysis_prompt(image_input)
    analysis_output, _, success = run_inference(model, processor, {'messages': analysis_messages}, max_new_tokens=100)
    if not success or not analysis_output:
        return {"error": "Failed to analyze the room image."}
    room_analysis = analysis_output[0]

    _, placement_messages = create_placement_prompt(room_type, style, image_input, room_analysis, furniture_config, style_materials)

    if important_prompt:
        for message in placement_messages:
            if message['role'] == 'user':
                for content_part in message['content']:
                    if content_part['type'] == 'text':
                        content_part['text'] += f"\n\n**Very Important Note:** {important_prompt}"
                        break
                break

    final_output, _, success = run_inference(model, processor, {'messages': placement_messages}, max_new_tokens=max_tokens)
    
    if success and final_output:
        return {"suggestion": final_output[0]}
    else:
        return {"error": "Failed to generate a final placement suggestion."}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})