import argparse
from pathlib import Path

from koboldapi import ImageProcessor, InstructTemplate, KoboldAPI

def process_image(api_url: str, image_path: Path, instruction: str):
    """ Process an image through the LLM with proper error handling.
    """
    api_client = KoboldAPI(api_url)
    processor = ImageProcessor(max_dimension=1024)
    wrapper = InstructTemplate(api_url)
    encoded_image, img_path = processor.process_image(str(image_path))
    if not encoded_image:
        print(f"Failed to process image: {img_path}")
        return
    
    prompt = wrapper.wrap_prompt(instruction) 
    return api_client.generate(
        prompt=prompt, 
        images=[encoded_image], 
        temperature=0
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