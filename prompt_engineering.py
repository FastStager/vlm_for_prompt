import random
from design_rules import RULES

def create_analysis_prompt(image_input):
    """Creates the prompt to analyze the visual properties of the room."""
    system_prompt = "You are a computer vision expert. Your task is to identify all permanent features of a room."
    user_prompt = "In one sentence, describe the room's unchangeable features: wall and floor color/material, and the exact locations of all windows, doors, and fireplaces. Be precise."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Analyze the provided image for permanent features.", messages

def create_placement_prompt(room_type, style, image_input, room_analysis, furniture_config, style_materials):
    """
    Creates a prompt with a strict hierarchy to prevent hallucinations while applying design rules.
    """
    if room_type in furniture_config:
        essential_furniture = ", ".join(furniture_config[room_type]["essential"])
        task_instruction = f"Your creative palette of essential furniture for a {room_type} includes: {essential_furniture}."
    else:
        print(f"Info: Room type '{room_type}' not in config. The VLM will infer essential items.")
        task_instruction = f"Use your knowledge to determine and incorporate essential furniture for a '{room_type}'."

    if style not in style_materials:
        style_details = "materials and colors appropriate for the style"
    else:
        style_info = style_materials[style]
        materials = style_info.get('materials', style_info.get('wood', 'appropriate materials'))
        colors = ", ".join(style_info.get('colors', ['cohesive colors']))
        style_details = f"Use materials like {materials} and a color palette of {colors}."
    
    rules_for_room = RULES.get(room_type, RULES["default"])
    rules_text = " ".join(rules_for_room)

    system_prompt = (
        f"You are an expert interior designer for a {style} {room_type}. "
        f"**Your absolute highest priority is to respect the existing room structure. You must NOT suggest placing furniture where it would block or contradict windows, doors, or fixed features identified in the room analysis.** "
        f"As a secondary guideline, creatively apply these design principles: '{rules_text}'. "
        "Your final output MUST be a single, detailed sentence following this narrative structure: "
        "'Place a [detailed material] [primary furniture item] [relative position], accompanied by a matching [secondary furniture item]. Add a [distinct object] [spatially related], and use a pair of identical [small lamps/decor] to create symmetry on [a surface]. Enhance with [ambient decor] to complement the [color palette or material].'"
    )
    
    user_prompt = (
        f"**Ground Truth - The Unchangeable Room:** \"{room_analysis}\"\n\n"
        f"Based on the unchangeable room features described above, create a furniture plan. "
        f"{task_instruction} {style_details} "
        "Ensure your layout perfectly fits the existing room without blocking any features. Follow the required sentence structure precisely."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "image", "image": image_input}, {"type": "text", "text": user_prompt}]}]
    return "Generate a rule-aware and non-hallucinatory furniture placement sentence.", messages