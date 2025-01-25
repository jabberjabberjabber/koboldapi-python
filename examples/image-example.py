import argparse
from pathlib import Path

from koboldapi import ImageProcessor, KoboldAPICore

def process_image(api_url: str, image_path: Path, instruction: str):
    """ Process an image through the LLM.
    """
    core = KoboldAPICore(api_url, temperature=0)
    processor = ImageProcessor(max_dimension=1024)
    encoded_image, img_path = processor.process_image(str(image_path))
    if not encoded_image:
        print(f"Failed to process image: {img_path}")
        return
    return core.wrap_and_generate(
        instruction, 
        images=[encoded_image]
    )

def main():
    parser = argparse.ArgumentParser(
        description="Process images through an LLM with custom instructions."
    )
    parser.add_argument(
        'image',
        type=Path,
        help='Image file to process'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:5001',
        help='KoboldCPP API URL'
    )
    parser.add_argument(
        '--instruction',
        default='Describe the image.',
        help='Instruction for the LLM'
    )
    args = parser.parse_args()

    try:
        result = process_image(args.api_url, args.image, args.instruction)
        if result:
            print(result)
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1
    return 0

if __name__ == '__main__':
    exit(main())