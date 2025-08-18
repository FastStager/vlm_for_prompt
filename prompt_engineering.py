from design_rules import RULES

def create_analysis_prompt(image_input):
    system_prompt = "You are a computer vision expert. Your task is to identify all permanent features of a room."
    user_prompt = "In one sentence, describe the room's unchangeable features: wall and floor color/material, and the exact locations of all windows, doors, and fireplaces. Be precise."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Analyze the provided image for permanent features.", messages

def create_placement_prompt(room_type, style, image_input, room_analysis, furniture_config, style_materials):
    if room_type in furniture_config:
        essential_furniture = ", ".join(furniture_config[room_type]["essential"])
        task_instruction = f"Your creative palette of essential furniture for a {room_type} includes: {essential_furniture}"
    else:
        task_instruction = f"Use your knowledge to determine and incorporate essential furniture for a '{room_type}'."

    if style not in style_materials:
        style_details = "materials and colors appropriate for the style"
    else:
        style_info = style_materials[style]
        materials = style_info.get('materials', style_info.get('wood', 'appropriate materials'))
        colors = ", ".join(style_info.get('colors', ['cohesive colors']))
        style_details = f"Use materials like {materials} and a color palette of {colors}."
    
    universal_rules = RULES.get("universal_absolute", [])
    specific_rules = RULES.get(room_type, RULES["default"])
    combined_rules = universal_rules + specific_rules
    rules_text = " ".join(combined_rules)

    system_prompt = (
        f"You are an expert interior designer specializing in {style} design. "
        f"Your primary function is to place furniture in a {room_type} based on an image analysis."
        "\n\n**CRITICAL SAFETY AND FUNCTIONALITY DIRECTIVE:**\n"
        "Your absolute, non-negotiable highest priority is to **NEVER block doors, entryways, or exits.** "
        "Placing any furniture, especially large items like a sofa, in front of a door is a critical failure. "
        "The room analysis provides the ground truth for door locations. You MUST adhere to it strictly."
        "\n\n**Design Guidelines:**\n"
        f"Apply these principles in your design: '{rules_text}'"
        "\n\n**Output Format:**\n"
        "Your final output MUST be a single, detailed sentence following this exact narrative structure: "
        "'Place a [detailed material] [primary furniture item] [relative position], accompanied by a matching [secondary furniture item]. Add a [distinct object] [spatially related], and use a pair of identical [small lamps/decor] to create symmetry on [a surface]. Enhance with [ambient decor] to complement the [color palette or material].'"
    )
    
    user_prompt = (
        f"**Ground Truth - Unchangeable Room Features:** \"{room_analysis}\"\n\n"
        f"**Task:** Based on the unchangeable room features described in the ground truth, create a furniture plan. "
        f"**CRITICAL REMINDER:** Do not place any items where they would obstruct the doors or entryways identified in the analysis. "
        f"Your task is to place the following: {task_instruction}. Use {style_details}. "
        "Ensure your layout perfectly fits the existing room. Follow the required sentence structure precisely."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Generate a rule-aware and non-hallucinatory furniture placement sentence.", messages