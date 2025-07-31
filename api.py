from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import torch
import json
from model_utils import setup_environment, load_model_adaptive, load_processor, run_inference, validate_image_url
from prompt_engineering import create_analysis_prompt, create_placement_prompt

app = FastAPI(title="AI Interior Designer API")
model = None
processor = None
config = None

class DesignRequest(BaseModel):
    room_type: str = "living room"
    style: str = "industrial"
    image_url: HttpUrl = "https://photos.zillowstatic.com/fp/3c83c384a192683219780302babe5ea9-p_f.jpg"
    max_tokens: int = 180
    important_prompt: str = ""

@app.on_event("startup")
async def startup_event():
    """Load models and config on server startup to handle 'cold start'."""
    global model, processor, config
    
    setup_environment()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("Loading model and processor...")
    model, _, _ = load_model_adaptive()
    processor = load_processor()
    print("Model and processor loaded.")

    print("Loading configuration...")
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        print("Configuration loaded.")
    except FileNotFoundError:
        raise RuntimeError("FATAL: config.json not found. The API cannot start.")

@app.post("/generate", summary="Generate Interior Design Suggestion")
async def generate_design(request: DesignRequest):
    """
    Receives design parameters, runs the full AI pipeline, and returns a furniture placement suggestion.
    """
    global model, processor, config
    
    try:
        validate_image_url(str(request.image_url))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Image Validation Error: {e}")

    furniture_config = config.get("FURNITURE_CONFIG", {})
    style_materials = config.get("STYLE_MATERIALS", {})

    _, analysis_messages = create_analysis_prompt(str(request.image_url))
    analysis_output, _, success = run_inference(model, processor, {'messages': analysis_messages}, max_new_tokens=100)
    if not success or not analysis_output:
        raise HTTPException(status_code=500, detail="Failed to analyze the room image.")
    room_analysis = analysis_output[0]

    _, placement_messages = create_placement_prompt(request.room_type, request.style, str(request.image_url), room_analysis, furniture_config, style_materials)

    if request.important_prompt:
        for message in placement_messages:
            if message['role'] == 'user':
                for content_part in message['content']:
                    if content_part['type'] == 'text':
                        content_part['text'] += f"\n\n**Very Important Note:** {request.important_prompt}"
                        break
                break

    final_output, _, success = run_inference(model, processor, {'messages': placement_messages}, max_new_tokens=request.max_tokens)
    
    if success and final_output:
        return {"suggestion": final_output[0]}
    else:
        raise HTTPException(status_code=500, detail="Failed to generate a final placement suggestion.")

if __name__ == "__main__":
    import uvicorn
    print("To run the local API server, use the command: uvicorn api:app --reload")
    uvicorn.run(app, host="127.0.0.1", port=8000)