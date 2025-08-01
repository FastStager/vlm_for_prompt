import random

def create_analysis_prompt(pil_image):
    """Creates the prompt to analyze the visual properties of the room."""
    system_prompt = "You are a computer vision expert analyzing an image of a room."
    user_prompt = "In one sentence, describe the room's key features, such as wall and floor color, materials, and any windows or doors."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": user_prompt}]}]
    return "Analyze the provided image.", messages

def create_placement_prompt(room_type, style, pil_image, room_analysis, furniture_config, style_materials):
    """Creates a detailed, context-aware prompt using a PIL image."""
    
    if room_type in furniture_config:
        essential_furniture = ", ".join(furniture_config[room_type]["essential"])
        task_instruction = f"Your creative palette of essential furniture for a {room_type} includes: {essential_furniture}. Weave these items naturally into your response."
    else:
        task_instruction = f"Use your knowledge to determine and incorporate essential furniture for a '{room_type}'."

    if style not in style_materials:
        style_details = "materials and colors appropriate for the style"
    else:
        style_info = style_materials[style]
        materials = style_info.get('materials', style_info.get('wood', 'appropriate materials'))
        colors = ", ".join(style_info.get('colors', ['cohesive colors']))
        style_details = f"Use materials like {materials} and a color palette of {colors}."

    system_prompt = (
        f"You are an expert interior designer creating a plan for a {style} {room_type}. "
        f"It is critical that every piece of furniture you select is appropriate for a {room_type}. "
        "Your response must be a single, detailed sentence of about 70-110 words, strictly following this narrative structure: "
        "'Place a [detailed material] [primary furniture item] [relative position], accompanied by a matching [secondary furniture item]. Add a [distinct object] [spatially related], and use a pair of identical [small lamps/decor] to create symmetry on [a surface]. Enhance with [ambient decor] to complement the [color palette or material].'"
    )
    
    user_prompt = (
        f"Room Analysis: \"{room_analysis}\".\n\n"
        f"Create a furniture plan based on this analysis. {task_instruction} {style_details} "
        "Adhere strictly to the required sentence structure and ensure the final output is detailed and contextually correct for the room type."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": user_prompt}]}]
    return "Generate a detailed and context-aware furniture placement sentence.", messages