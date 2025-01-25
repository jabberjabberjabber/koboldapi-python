# KoboldAPI

A Python library for interacting with KoboldCPP APIs, providing high-level abstractions for text processing, image handling, and generation tasks.

## Features

- **Text Processing**
  - Intelligent text chunking 
  - Streaming generation capabilities
  - Token counting and management

- **Image Processing**
  - Support for multiple image formats (JPEG, PNG, GIF, TIFF, WEBP, HEIF, RAW)
  - Automatic image resizing and optimization
  - RAW image processing with thumbnail extraction
  - Base64 encoding for API transmission

- **Template Management**
  - Flexible template system for different LLM models
  - Custom template directory support
  - Built-in default templates
  - Jinja2 templating integration

- **API Integration**
  - Robust error handling
  - Connection management
  - Streaming support
  - Comprehensive API endpoint coverage

## Installation

```bash
pip install koboldapi
```

## Quick Start

### Text Processing Example

```python
from koboldapi import KoboldAPICore, ChunkingProcessor

# Initialize the API client
core = KoboldAPICore(api_url="http://localhost:5001")

# Create a chunking processor
chunker = ChunkingProcessor(core.api_client, max_chunk_length=2048)

# Process text
chunks, metadata = chunker.chunk_file("document.txt")
for chunk, token_count in chunks:
    response = core.wrap_and_generate(
        instruction="Summarize this text:",
        content=chunk
    )
    print(response)
```

### Image Processing Example

```python
from koboldapi import KoboldAPICore, ImageProcessor

# Initialize processors
core = KoboldAPICore(api_url="http://localhost:5001")
processor = ImageProcessor(max_dimension=1024)

# Process image
encoded_image, img_path = processor.process_image("image.jpg")
response = core.wrap_and_generate(
    instruction="Describe this image:",
    images=[encoded_image]
)
print(response)
```

### Streaming Generation Example

```python
from koboldapi import KoboldAPICore
import asyncio

async def stream_example():
    core = KoboldAPICore(api_url="http://localhost:5001")
    
    # Stream tokens as they're generated
    async for token in core.api_client.stream_generate(
        prompt="Write a story about a robot:",
        max_length=200
    ):
        print(token, end='', flush=True)
    
    # Or collect all tokens into final result
    result = await core.api_client.generate_sync(
        prompt="Write a story about a robot:",
        max_length=200
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(stream_example())
	
```

## Detailed Documentation

### KoboldAPICore

The main interface for interacting with KoboldCPP APIs.

```python
core = KoboldAPICore(
    api_url="http://localhost:5001",
    api_password=None,  # Optional API password
    generation_params={  # Optional generation parameters
        'temp': 0.7,
        'top_k': 40,
        'top_p': 0.9
    },
    templates_directory="path/to/templates"  # Optional custom templates
)
```

### Text Processing

The ChunkingProcessor class handles text segmentation and processing:

```python
chunker = ChunkingProcessor(
    api_client,  # KoboldAPI instance
    max_chunk_length=2048,  # Maximum tokens per chunk
    max_total_chunks=1000  # Maximum number of chunks
)

# Process a file
chunks, metadata = chunker.chunk_file("document.txt")

# Process raw text
chunks = chunker.chunk_text("Your text content here")
```

### Image Processing

The ImageProcessor class handles image preparation and optimization:

```python
processor = ImageProcessor(
    max_dimension=1024,  # Maximum image dimension
    patch_sizes=[8, 14, 16, 32],  # Patch size options
    max_file_size=50 * 1024 * 1024  # Maximum file size in bytes
)

# Process an image
encoded_image, path = processor.process_image("image.jpg")
```

### Template Management

Custom templates can be provided in JSON format:

```json
{
    "template_name": {
        "name": ["model_name_pattern"],
        "system_start": "\nSystem: ",
        "system_end": "\n",
        "user_start": "User: ",
        "user_end": "\n",
        "assistant_start": "Assistant: "
    }
}
```

## Command Line Tools

The package includes example scripts for common tasks:

### Text Processing Script

```bash
python text-example.py input.txt \
    --task translate \
    --api-url http://localhost:5001 \
    --language French \
    --max-chunk-size 8192
```

Available tasks:
- translate: Translate text to specified language
- summary: Generate text summary
- correct: Fix grammar and spelling
- distill: Create concise version

### Image Processing Script

```bash
python image-example.py image.jpg \
    --api-url http://localhost:5001 \
    --instruction "Describe the image in detail."
```

## Error Handling

The library provides custom exceptions for error handling:

```python
try:
    result = core.wrap_and_generate(instruction="Your instruction")
except KoboldAPIError as e:
    print(f"API Error: {e}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.
