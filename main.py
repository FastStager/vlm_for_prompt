import os
import gc
import json
import argparse
import torch
from model_utils import setup_environment, load_model_adaptive, load_processor, run_inference, validate_and_process_image_input
from prompt_engineering import create_analysis_prompt, create_placement_prompt

def load_config(filepath="config.json"):
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
        image_input = validate_and_process_image_input(image_url=args.image_url)
    except ValueError as e:
        print(f"\n[FATAL] Image Validation Error: {e}"); exit()

    setup_environment()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    config = load_config()
    furniture_config = config.get("FURNITURE_CONFIG", {}); style_materials = config.get("STYLE_MATERIALS", {})

    try:
        model, _, _ = load_model_adaptive()
        processor = load_processor()

        print("\nAnalyzing room image...")
        # Pass the validated image_input to the prompt function
        _, analysis_messages = create_analysis_prompt(image_input)
        analysis_output, _, success = run_inference(model, processor, {'messages': analysis_messages}, max_new_tokens=100)
        
        if not success or not analysis_output: raise RuntimeError("Failed to analyze the room image.")
        
        room_analysis = analysis_output[0]
        print(f"Analysis complete: {room_analysis}")

        print("\nGenerating furniture placement...")
        _, placement_messages = create_placement_prompt(args.room_type, args.style, image_input, room_analysis, furniture_config, style_materials)
        
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
    parser = argparse.ArgumentParser(description="AI for Interior Furniture Placement", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--room_type", type=str, default="living room", help="Type of the room to design.") 
    parser.add_argument("--style", type=str, default="industrial", help="Desired interior design style.") 
    parser.add_argument("--image_url", type=str, default="https://photos.zillowstatic.com/fp/3c83c384a192683219780302babe5ea9-p_f.jpg", help="URL of the empty room image.")
    args = parser.parse_args()
    main(args)