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
        "You are a Spatial Layout AI that generates a single, machine-readable instruction for an image editing model. Your output must be one complete sentence that is both spatially intelligent and fully comprehensive."
        "\n\n**1. PLACEMENT STRATEGY LOGIC (Highest Priority):**\n"
        "-   **IDENTIFY CONFLICT:** The sofa must face the focal point (fireplace/main window). Is the door on the same wall or an adjacent wall to the focal point? If so, this is a conflict.\n"
        "-   **RESOLVE CONFLICT:**\n"
        "    -   **No Conflict:** Place the sofa against the wall opposite the focal point.\n"
        "    -   **Conflict Exists:** Place the sofa **perpendicular** to the focal point wall to create a clear walkway from the door. This is a mandatory rule.\n"
        "\n**2. COMPLETENESS REQUIREMENT (Equally High Priority):**\n"
        "-   You will be given a 'Placement Checklist' of essential furniture.\n"
        "-   Your final, single-sentence output **MUST** include placement instructions for **EVERY** item on that list. Do not stop after placing just the sofa.\n"
        "\n**EXAMPLE OF A COMPLETE, CORRECT OUTPUT:**\n"
        "-   **Ground Truth:** '...a door on the left and a fireplace on the right of the back wall.'\n"
        "-   **Checklist:** 'sofa, coffee table, tv stand'.\n"
        "-   **Your Correct Output:** 'Place a reclaimed wood sofa perpendicular to the back wall to create a clear path from the door, add a matching coffee table in front of it, and position a black metal TV stand on the opposite wall.'"
        "\n\n**TEMPLATE:** \"Place a [main item] [safe/oriented position], a [second item] [relative position], and a [third item] [final position].\""
    )
    
    user_prompt = (
        f"**Ground Truth:** \"{room_analysis}\"\n\n"
        f"**Task:** Generate ONE complete sentence using the Placement Strategy and the Checklist below.\n\n"
        f"**- Critical Logic:** First, determine if the door and focal point conflict. If so, use the perpendicular placement strategy. The path from the door MUST be clear.\n"
        f"**- Placement Checklist:** Your single sentence MUST include placement instructions for every item on this list: **{essential_furniture}**. Do not omit any.\n"
        f"**- Style:** {style} (materials: {materials}; colors: {colors}).\n\n"
        "Generate the complete and spatially correct command string now."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Generate a rule-aware and non-hallucinatory furniture placement sentence.", messages