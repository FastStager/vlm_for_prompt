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
        "You are a Spatial Layout AI that generates a single, machine-readable instruction for an image editing model. Your primary directive is to maintain a clear, unobstructed path from all doors and entryways."
        "\n\n**PLACEMENT STRATEGY LOGIC (Follow this hierarchy):**\n"
        "1.  **IDENTIFY CONFLICT:** The main furniture (sofa/bed) should face the room's focal point (fireplace/main window). Is the door located on the same wall or a wall adjacent to the focal point? If so, this is a conflict.\n"
        "2.  **RESOLVE CONFLICT:**\n"
        "    -   **If NO CONFLICT:** Place the sofa against the wall opposite the focal point. (e.g., 'Place a sofa against the left wall, facing the fireplace on the right wall.')\n"
        "    -   **If there IS A CONFLICT:** You MUST place the sofa **perpendicular** to the focal point wall. This creates a clear walkway from the door. Your instruction must be explicit about this orientation. (e.g., 'Place a sofa perpendicular to the fireplace wall to create an open walkway from the adjacent door.')\n"
        "3.  **FINAL COMMAND:** Build the rest of the command around this primary, safe placement."
        "\n\n**EXAMPLE OF CONFLICT RESOLUTION:**\n"
        "-   **Ground Truth:** '...a door on the left side of the back wall and a fireplace on the right side of the same back wall.'\n"
        "-   **Your Correct Output:** 'Place a reclaimed wood sofa perpendicular to the back wall, facing the fireplace, leaving a wide path from the door. Add a coffee table in front of the sofa and a TV stand on the opposite wall.'"
        "\n\n**TEMPLATE:** \"Place a [material] [furniture] [SPECIFIC, SAFE, and ORIENTED position], a [second item] [relative position], and a [third item] [near another feature].\""
    )
    
    user_prompt = (
        f"**Ground Truth:** \"{room_analysis}\"\n\n"
        f"**Task:** Generate a single command string using the PLACEMENT STRATEGY LOGIC.\n\n"
        f"**- Critical Instruction:** Analyze if the door and focal point conflict. If they do, you MUST use a perpendicular placement to guarantee a clear path from the door.\n"
        f"**- Furniture:** {essential_furniture}.\n"
        f"**- Style:** {style} (materials: {materials}; colors: {colors}).\n\n"
        "Generate the specific command string now. Your instruction for the sofa's position and orientation is the most important part."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Generate a rule-aware and non-hallucinatory furniture placement sentence.", messages