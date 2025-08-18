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
        "You are a Spatial Reasoning AI. Your sole purpose is to generate a single, machine-readable instruction string for an image editing model. Your reasoning must be explicit in your output."
        "\n\n**CORE LOGIC: SPATIAL BOUNDARY AWARENESS**\n"
        "1.  **Identify No-Go Zones:** From the 'Ground Truth' analysis, identify the exact location of all doors and entryways. These are forbidden areas.\n"
        "2.  **Use Safe Anchors:** Place the largest furniture item (e.g., sofa, bed) first. You MUST position it relative to a permanent, safe feature (like a window, a fireplace, or a wall explicitly opposite the door).\n"
        "3.  **Generate Spatially-Aware Instructions:** Your output instruction string MUST use language that is unambiguous and inherently respects the No-Go Zones. Your output is your proof of reasoning."
        "\n\n**Example of Bad vs. Good Instructions:**\n"
        "-   **BAD (Ambiguous):** `Place a sofa against the far wall.` (The door could be on that wall).\n"
        "-   **GOOD (Spatially-Aware):** `Place a sofa against the wall opposite the main door.` (Unambiguous and safe).\n"
        "-   **GOOD (Spatially-Aware):** `Place a bed between the two windows on the left wall, leaving the doorway on the right wall clear.` (Uses multiple anchors and explicitly states clearance)."
        "\n\n**OUTPUT TEMPLATE (Follow Precisely):**\n"
        "\"Place a [material] [main furniture] [spatially-aware position referencing safe anchors], a [second item] [position relative to the first item or another anchor], and a [third item] [final position]. Decorate with [decor elements] on [a surface].\""
        "\n\n**FORMATTING RULES:**\n"
        "-   Start your response *directly* with 'Place'. No preamble.\n"
        "-   The entire output must be one single sentence."
    )
    
    user_prompt = (
        f"**Ground Truth (Unchangeable Room):** \"{room_analysis}\"\n\n"
        f"**Task:** Generate a single-line instruction string for a {style} {room_type}.\n\n"
        f"**CRITICAL:** Your instruction string must be spatially coherent and explicitly avoid the no-go zones identified in the Ground Truth. Use the 'Good Instruction' examples as your guide for phrasing.\n\n"
        f"**- Furniture to Use:** {essential_furniture}.\n"
        f"**- Style:** {style} (materials like {materials}; colors like {colors}).\n\n"
        "Generate the instruction string now."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Generate a rule-aware and non-hallucinatory furniture placement sentence.", messages