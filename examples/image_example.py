import argparse
import base64
from pathlib import Path
from typing import Optional, List, Tuple
import sys

from koboldapi import KoboldAPICore

def encode_image(file_path: Path) -> Optional[str]:
    """Encode image file to base64."""
    try:
        return base64.b64encode(file_path.read_bytes()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding {file_path}: {e}", file=sys.stderr)
        return None

def process_image(core: KoboldAPICore, image_path: Path, 
                 instruction: str) -> Tuple[Optional[str], Path]:
    """Process a single image through the LLM."""
    if not image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        print(f"Unsupported file type: {image_path}", file=sys.stderr)
        return None, image_path
        
    # Encode image
    encoded = encode_image(image_path)
    if not encoded:
        return None, image_path
        
    max_context = core.api_client.get_max_context_length()
    
    # Generate text from image
    try:
        prompt = core.template_wrapper.wrap_prompt(  # Changed from template to template_wrapper
            instruction=instruction,
            system_instruction="You are a helpful assistant."
        )[0]
        
        result = core.api_client.generate(  # Changed from api to api_client
            prompt=prompt,
            images=[encoded],
            temperature=0,
            top_p=1,
            top_k=0,
            rep_pen=1.05,
            max_length=max_context // 2
        )
        
        return result, image_path.with_suffix('.txt')
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}", file=sys.stderr)
        return None, image_path

def process_images(config: dict, image_paths: List[Path], 
                  instruction: str) -> int:
    """Process multiple images through the LLM."""
    core = KoboldAPICore(config)
    had_error = False
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"Processing {image_path.name} ({i}/{len(image_paths)})...")
        
        result, output_path = process_image(core, image_path, instruction)
        if result:
            try:
                output_path.write_text(result, encoding='utf-8')
                print(f"Saved output to {output_path}")
            except Exception as e:
                print(f"Error saving to {output_path}: {e}", file=sys.stderr)
                had_error = True
        else:
            had_error = True
            
    return 1 if had_error else 0

def main():
    parser = argparse.ArgumentParser(description="Send images to the LLM with a prompt.")
    
    parser.add_argument('images', nargs='+', help='Image files to process')
    parser.add_argument('--api-url', default='http://localhost:5001',
                       help='KoboldCPP API URL')
    parser.add_argument('--templates', default='templates',
                       help='Templates directory path')
    parser.add_argument('--instruction', 
                       default='Describe the image in detail.',
                       help='Instruction for the LLM')
    
    args = parser.parse_args()
    config_dict = {
        "api_url": args.api_url,
        "api_password": "",  
        "templates_directory": args.templates,
    }

    # Convert paths
    image_paths = [Path(p) for p in args.images]
    
    # Process images
    return process_images(config_dict, image_paths, args.instruction)

if __name__ == '__main__':
    exit(main())
