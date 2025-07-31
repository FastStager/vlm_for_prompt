# AI Interior Designer

This project uses the Qwen2-VL model to provide AI-powered interior design suggestions. It analyzes an image of an empty room and, based on user-selected styles and room types, generates a furniture placement recommendation.

### Core Features

- **Room Analysis**: Identifies permanent features like windows, doors, and walls.
- **Style-Based Generation**: Applies design rules for various styles (e.g., Industrial, Scandinavian).
- **Customizable**: Allows tweaking of output length (`max_tokens`) and adding specific user requests.
- **Serverless Ready**: Includes a `runpod_handler.py` for deployment on RunPod.

### How to Run Locally

**1. Install Dependencies:**

Ensure you have Python 3.10+ and `pip` installed.

```bash
pip install -r requirements.txt
```

**2. Start the Local API Server:**

This server hosts the model and logic. It needs to be running before you can use the Gradio UI.

```bash
uvicorn api:app --reload
```

**3. Launch the Gradio UI:**

In a **new terminal**, run the following command to start the user interface.

```bash
python ui.py
```
