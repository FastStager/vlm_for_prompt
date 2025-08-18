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
        style_details = "appropriate materials and colors"
        materials = "appropriate materials"
        colors = "a cohesive palette"
    else:
        style_info = style_materials[style]
        materials = style_info.get('materials', style_info.get('wood', 'appropriate materials'))
        colors = ", ".join(style_info.get('colors', ['cohesive colors']))
        style_details = f"materials like {materials} and a color palette of {colors}"

    # The critical rule remains the top priority.
    universal_rules = RULES.get("universal_absolute", [])
    rules_text = " ".join(universal_rules)

    system_prompt = (
        "You are a specialized AI that generates precise, machine-readable layout instructions for an image editing model. "
        "Your output is not for humans; it is a direct command for another AI. "
        "It must be concise, descriptive, and spatially accurate."
        "\n\n**CRITICAL DIRECTIVE:**\n"
        f"Your absolute, non-negotiable highest priority is to follow this rule: '{rules_text}'. "
        "Placing furniture that blocks doors or exits is a fatal error. The room analysis is your ground truth."
        "\n\n**OUTPUT TEMPLATE (MUST be followed exactly):**\n"
        "\"Place a [material] [furniture item] [relative position], a [color] [object] [spatially related to first], and [another object] [near/by/next to some anchor]. Add [small decor elements] that match [color palette or anchor element].\""
        "\n\n**FORMATTING RULES:**\n"
        "- Start your response *directly* with the word 'Place'. No introductory phrases or explanations.\n"
        "- The entire output must be a single, unbroken sentence.\n"
        "- Be descriptive with materials and colors, but concise with locations (e.g., 'against the back wall', 'centered under the window', 'to the left of the fireplace')."
    )
    
    user_prompt = (
        f"**Ground Truth (Unchangeable Room):** \"{room_analysis}\"\n\n"
        f"**Task:** Generate a single-line instruction string for a {style} {room_type} using the required template.\n"
        f"**CRITICAL REMINDER:** Your layout MUST respect all doors and entryways from the analysis. Do not obstruct them.\n\n"
        f"**- Furniture to Use:** {essential_furniture}.\n"
        f"**- Style:** {style} (use {style_details}).\n\n"
        "Generate the instruction string now."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Generate a rule-aware and non-hallucinatory furniture placement sentence.", messages