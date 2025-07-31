import gradio as gr
import requests
import json

try:
    with open("config.json", 'r') as f:
        config = json.load(f)
    room_types = list(config.get("FURNITURE_CONFIG", {}).keys())
    styles = list(config.get("STYLE_MATERIALS", {}).keys())
except FileNotFoundError:
    print("Warning: config.json not found. Using default lists for UI.")
    room_types = ["living room", "bedroom", "kitchen", "bathroom"]
    styles = ["scandinavian", "japandi", "modern", "minimalist", "industrial"]

API_URL = "http://127.0.0.1:8000/generate"

def get_ai_suggestion(room_type, style, image_url, max_tokens, important_prompt):
    """
    Sends a POST request to the local FastAPI server and returns the AI suggestion.
    """
    if not image_url:
        return "Error: Please provide an image URL."
    
    payload = {
        "room_type": room_type,
        "style": style,
        "image_url": image_url,
        "max_tokens": int(max_tokens),
        "important_prompt": important_prompt
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300) 
        response.raise_for_status() 
        
        result = response.json()
        return result.get("suggestion", "Error: No 'suggestion' key found in API response.")

    except requests.exceptions.HTTPError as http_err:
        try:
            detail = http_err.response.json().get("detail", str(http_err))
        except json.JSONDecodeError:
            detail = http_err.response.text
        return f"HTTP Error: {detail}"
    except requests.exceptions.RequestException as req_err:
        return f"API Connection Error: {req_err}\n\nIs the local API server running? In your terminal, run:\nuvicorn api:app --reload"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.cyan)) as demo:
    gr.Markdown(
        """
        # 🛋️ AI Interior Designer
        Provide a picture of an empty room and select your desired style. The AI will analyze the room's permanent 
        features and suggest a furniture layout based on established design principles.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_url_textbox = gr.Textbox(
                label="Image URL",
                placeholder="https://your-image-url.com/room.jpg",
                value="https://photos.zillowstatic.com/fp/3c83c384a192683219780302babe5ea9-p_f.jpg"
            )
            room_type_dropdown = gr.Dropdown(label="Room Type", choices=room_types, value="living room")
            style_dropdown = gr.Dropdown(label="Design Style", choices=styles, value="industrial")
            
            gr.Markdown("---")
            
            max_tokens_slider = gr.Slider(
                label="Max Output Tokens", minimum=50, maximum=512, step=1, value=180,
                info="Controls the maximum length of the generated description."
            )
            important_prompt_textbox = gr.Textbox(
                label="Important Details to Include",
                placeholder="(Optional) e.g., 'Make sure to include a reading corner near the window.'",
                lines=3
            )
            submit_button = gr.Button("Generate Design ✨", variant="primary")
            
        with gr.Column(scale=2):
            image_preview = gr.Image(label="Room Preview", type="filepath", value="https://photos.zillowstatic.com/fp/3c83c384a192683219780302babe5ea9-p_f.jpg")
            output_textbox = gr.Textbox(label="AI Suggestion", lines=10, interactive=False)
            
    image_url_textbox.change(lambda x: x, inputs=image_url_textbox, outputs=image_preview)
    
    submit_button.click(
        fn=get_ai_suggestion,
        inputs=[room_type_dropdown, style_dropdown, image_url_textbox, max_tokens_slider, important_prompt_textbox],
        outputs=output_textbox
    )

if __name__ == "__main__":
    print("Launching Gradio UI...")
    demo.launch()