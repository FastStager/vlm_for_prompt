import os
import gc
import json
import argparse
import torch
from model_utils import setup_environment, load_model_adaptive, load_processor, run_inference, load_and_validate_image
from prompt_engineering import create_analysis_prompt, create_placement_prompt

def load_config(filepath="config.json"):
    """Loads configuration from a JSON file."""
    try:
        with open(filepath, 'r') as f: return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{filepath}'. Exiting."); exit()

def main(args):
    """Main function to run the interior design AI."""
    print("\n=== Intelligent Furniture Placement AI ===")
    print(f"Goal: Designing a {args.style.title()} {args.room_type.title()}.")
    print("-----------------------------------------")
    
    try:
        print("Loading and validating image source...")
        if args.image_url:
            pil_image = load_and_validate_image(args.image_url, source_type='url')
        elif args.image_file:
            pil_image = load_and_validate_image(args.image_file, source_type='file')
        elif args.image_base64:
            pil_image = load_and_validate_image(args.image_base64, source_type='base64')
        print("Image successfully loaded and validated.")

    except ValueError as e:
        print(f"\n[FATAL] Image Error: {e}"); exit()

    setup_environment()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    config = load_config()
    furniture_config = config.get("FURNITURE_CONFIG", {}); style_materials = config.get("STYLE_MATERIALS", {})

    try:
        model, _, _ = load_model_adaptive()
        processor = load_processor()

        print("\nAnalyzing room image...")
        _, analysis_messages = create_analysis_prompt(pil_image)
        analysis_output, _, success = run_inference(model, processor, {'messages': analysis_messages}, max_new_tokens=100)
        
        if not success or not analysis_output: raise RuntimeError("Failed to analyze the room image.")
        room_analysis = analysis_output[0]
        print(f"Analysis complete: {room_analysis}")

        print("\nGenerating furniture placement...")
        _, placement_messages = create_placement_prompt(args.room_type, args.style, pil_image, room_analysis, furniture_config, style_materials)
        final_output, _, success = run_inference(model, processor, {'messages': placement_messages}, max_new_tokens=180)
        
        if success and final_output:
            print("\n======================================"); print("AI Interior Designer Suggestion:"); print("======================================")
            print(final_output[0])
            print("======================================\n")
        else:
            print("\nFailed to generate a final placement suggestion.")

    except Exception as e:
        print(f"\nAn unexpected runtime error occurred: {e}")
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print("Script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Interior Designer. Provide one image source.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_url", type=str, help="URL of the empty room image.")
    group.add_argument("--image_file", type=str, help="Local file path of the empty room image.")
    group.add_argument("--image_base64", type=str, help="Base64 encoded string of the image.")
    
    parser.add_argument("--room_type", type=str, default="living room", help="Type of the room to design.")
    parser.add_argument("--style", type=str, default="industrial", help="Desired interior design style.")
    
    args = parser.parse_args()
    main(args)