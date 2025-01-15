import argparse
from pathlib import Path
import sys

from koboldapi import KoboldAPICore, ImageProcessor



def process_images(config: dict, image_paths: list[Path], instruction: str,
                  output_dir: Path | None = None) -> int:
    """ Process multiple images through the LLM using ImageProcessor.
    
        Args:
            config: Configuration dictionary for KoboldAPICore
            image_paths: List of paths to image files
            instruction: Instruction for the LLM
            output_dir: Optional directory to save results
            
        Returns:
            Exit code (0 for success, 1 for errors)
    """
    core = KoboldAPICore(config)
    processor = ImageProcessor(core)
  
    results = processor.process_batch(
        image_paths,
        instruction=instruction,
        output_dir=output_dir,
        temperature=0.1,
        top_p=1.0,
        top_k=0,
        rep_pen=1.05
    )
    
    return 1 if results["metadata"]["had_errors"] else 0


def main():
    """ Main entry point with argument parsing. """
    parser = argparse.ArgumentParser(
        description="Process images through an LLM with custom instructions."
    )
    
    parser.add_argument(
        'images',
        nargs='+',
        help='Image files to process'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:5001',
        help='KoboldCPP API URL'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Directory to save results (optional)'
    )
    parser.add_argument(
        '--instruction',
        default='Describe the image in detail.',
        help='Instruction for the LLM'
    )
    
    args = parser.parse_args()

    config_dict = {
        "api_url": args.api_url,
        "api_password": "",
    }
    image_paths = [Path(p) for p in args.images]
    
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    return process_images(
        config_dict,
        image_paths,
        args.instruction,
        args.output_dir
    )


if __name__ == '__main__':
    exit(main())
