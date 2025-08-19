from design_rules import RULES

def create_analysis_prompt(image_input):
    system_prompt = "You are a computer vision expert. Your task is to identify all permanent features of a room."
    user_prompt = "In one sentence, describe the room's unchangeable features: wall and floor color/material, and the exact locations of all windows, doors, and fireplaces. Be precise."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Analyze the provided image for permanent features.", messages

def create_placement_prompt(room_type, style, image_input, room_analysis, furniture_config, style_materials):
    if room_type in furniture_config:
        essential_furniture = ", ".join(furniture_config[room_type]["essential"])
    else:
        essential_furniture = f"essential furniture for a '{room_type}'"

    if style not in style_materials:
        materials = "appropriate materials"
        colors = "a cohesive palette"
    else:
        style_info = style_materials[style]
        materials = style_info.get('materials', style_info.get('wood', 'appropriate materials'))
        colors = ", ".join(style_info.get('colors', ['cohesive colors']))

    system_prompt = (
        "You are an AI that generates a single, direct instruction string for an image editing model. Your output must be spatially precise and follow a strict logical process."
        "\n\n**CORE LOGIC:**\n"
        "1.  **Identify the Door:** Read the 'Ground Truth' analysis to find which wall the door is on (e.g., 'right wall', 'back wall').\n"
        "2.  **Select the Opposite Wall:** Identify the wall directly opposite the door. This is the **only** valid location for the main sofa or bed.\n"
        "3.  **Generate the Command:** Create a single-sentence command placing the main furniture on that specific, valid wall. All other furniture is placed relative to it."
        "\n\n**EXAMPLE:**\n"
        "-   **Ground Truth Input:** 'The room has a single door on the right wall and a window on the back wall.'\n"
        "-   **Your Correct Output:** 'Place a reclaimed wood sofa against the left wall, a metal coffee table in front of it, and a TV stand against the back wall under the window. Add two small lamps on the TV stand.'"
        "\n\n**OUTPUT TEMPLATE:**\n"
        "\"Place a [material] [furniture] [at the specific location that avoids the door], a [second item] [relative to the first], and a [third item] [near another feature]. Add [decor] on [a surface].\""
    )
    
    user_prompt = (
        f"**Ground Truth:** \"{room_analysis}\"\n\n"
        f"**Task:** Using the CORE LOGIC, generate a single, unambiguous command string for a {style} {room_type}.\n\n"
        f"**- Critical Instruction:** Find the wall with the door in the Ground Truth. Your command MUST place the sofa on the wall OPPOSITE to it.\n"
        f"**- Furniture:** {essential_furniture}.\n"
        f"**- Style:** {style} (materials like {materials}; colors like {colors}).\n\n"
        "Generate the command string now. Be specific and avoid vague terms."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Generate a rule-aware and non-hallucinatory furniture placement sentence.", messages