import argparse
import base64
from pathlib import Path
from typing import Optional, List, Tuple
import sys

from koboldapi import LLMConfig, LLMToolsCore
from koboldapi.core.core import LLMToolsCore

# recommended models:
# minicpm-v-2.6, qvq, qwen2vl

def encode_image(file_path: Path) -> Optional[str]:
    """Encode image file to base64.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Base64 encoded string or None if failed
    """
    try:
        return base64.b64encode(file_path.read_bytes()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding {file_path}: {e}", file=sys.stderr)
        return None

def process_image(core: LLMToolsCore, image_path: Path, 
                 instruction: str) -> Tuple[Optional[str], Path]:
    """Process a single image through the LLM.
    
    Args:
        core: LLMToolsCore instance
        image_path: Path to image file
        instruction: Instruction for the LLM
        
    Returns:
        Tuple of (result text, output path) or (None, path) if failed
    """
    if not image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        print(f"Unsupported file type: {image_path}", file=sys.stderr)
        return None, image_path
        
    # Encode image
    encoded = encode_image(image_path)
    if not encoded:
        return None, image_path
        
    # Generate text from image
    try:
        prompt = core.template.wrap_prompt(
            instruction=instruction,
            system_instruction="You are an OCR system. Extract text exactly as shown."
        )[0]
        
        result = core.api.generate(
            prompt=prompt,
            images=[encoded],
            temperature=0,
            top_p=1,
            top_k=0,
            rep_pen=1.05
        )
        
        return result, image_path.with_suffix('.txt')
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}", file=sys.stderr)
        return None, image_path

def process_images(image_paths: List[Path], config: LLMConfig, 
                  instruction: str) -> int:
    """Process multiple images through the LLM.
    
    Args:
        image_paths: List of image file paths
        config: LLMConfig instance 
        instruction: Instruction for the LLM
        
    Returns:
        Exit code (0 for success, 1 for any failures)
    """
    core = LLMToolsCore(config)
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
    parser = argparse.ArgumentParser(description="Extract text from images using LLM")
    
    parser.add_argument('images', nargs='+', help='Image files to process')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--api-url', default='http://localhost:5001',
                       help='KoboldCPP API URL')
    parser.add_argument('--templates', default='templates',
                       help='Templates directory path')
    parser.add_argument('--instruction', 
                       default='Repeat verbatim all text shown in the image.',
                       help='Instruction for the LLM')
    
    args = parser.parse_args()
    
    try:
        # Set up configuration
        if args.config:
            config = LLMConfig.from_json(args.config)
        else:
            config = LLMConfig(
                api_url=args.api_url,
                api_password="",
                templates_directory=args.templates
            )
            
        # Convert paths
        image_paths = [Path(p) for p in args.images]
        
        # Process images
        return process_images(image_paths, config, args.instruction)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    exit(main())