# KoboldAPI

A Python library for interacting with KoboldCPP API, providing robust text and image processing capabilities with built-in chunking and templating support.

## Features

- Text and document processing with Apache Tika integration
- Smart chunking with regex-based natural language boundaries
- Image processing with support for various formats including RAW
- Chat templating system supporting multiple LLM formats
- Async and streaming generation capabilities
- Built-in error handling and retry mechanisms

## Installation

```bash
pip install koboldapi
```

Required dependencies for full functionality:
```bash
pip install extractous pillow-heif rawpy
```

## Quick Start

### Basic API Usage

```python
from koboldapi import KoboldAPICore

# Initialize the core API client
api = KoboldAPICore("http://localhost:5001")

# Simple generation
response = api.api_client.generate("What is the capital of France?")
print(response)

# Get model info
model_info = api.get_model_info()
print(f"Model: {model_info['name']}")
print(f"Context Length: {model_info['context_length']}")
```

### Processing Text Documents

```python
import asyncio
from pathlib import Path
from koboldapi import KoboldAPICore, ChunkingProcessor

async def process_document(file_path: str):
    # Initialize API
    core = KoboldAPICore("http://localhost:5001")
    
    # Create processor with 2048 token chunks
    processor = ChunkingProcessor(core.api_client, max_chunk_length=2048)
    
    # Process document - works with any format supported by Apache Tika
    chunks, metadata = processor.chunk_file(Path(file_path))
    
    print(f"Document metadata: {metadata}")
    print(f"Number of chunks: {len(chunks)}")
    
    # Process each chunk
    for i, (chunk, token_count) in enumerate(chunks):
        response = core.api_client.generate(
            prompt=chunk,
            max_length=1024
        )
        print(f"\nChunk {i+1} ({token_count} tokens):")
        print(response)

# Run with asyncio
asyncio.run(process_document("document.pdf"))
```

### Image Processing

```python
from koboldapi import ImageProcessor, InstructTemplate, KoboldAPI

def analyze_image(image_path: str, instruction: str = "Describe this image"):
    # Initialize components
    api = KoboldAPI("http://localhost:5001")
    processor = ImageProcessor(max_dimension=1024)
    template = InstructTemplate("http://localhost:5001")
    
    # Process image
    encoded_image, _ = processor.process_image(image_path)
    if not encoded_image:
        raise ValueError("Failed to process image")
    
    # Create prompt with template
    prompt = template.wrap_prompt(instruction)
    
    # Generate description
    return api.generate(
        prompt=prompt,
        images=[encoded_image],
        temperature=0.7
    )

# Use the function
result = analyze_image("photo.jpg", "What objects do you see in this image?")
print(result)
```

### Chat with Context

```python
from koboldapi import KoboldAPICore

class ChatSession:
    def __init__(self, api_url: str):
        self.core = KoboldAPICore(api_url)
        self.template = self.core.template_wrapper
        self.history = []
        
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        
    def get_response(self, user_input: str) -> str:
        # Add user message to history
        self.add_message("user", user_input)
        
        # Build context from history
        context = ""
        for msg in self.history[-5:]:  # Keep last 5 messages for context
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            else:
                context += f"Assistant: {msg['content']}\n"
                
        # Generate response using template
        prompt = self.template.wrap_prompt(
            instruction="Respond to the user's message",
            content=context
        )
        
        response = self.core.api_client.generate(prompt)
        self.add_message("assistant", response)
        
        return response

# Example usage
chat = ChatSession("http://localhost:5001")
response = chat.get_response("Tell me about neural networks")
print(response)
```

### Advanced Features

#### Streaming Generation

```python
import asyncio
from koboldapi import KoboldAPI

async def stream_response(prompt: str):
    api = KoboldAPI("http://localhost:5001")
    
    async for token in api.stream_generate(prompt):
        print(token, end='', flush=True)
    print()

# Use with asyncio
asyncio.run(stream_response("Write a short story about a robot"))
```

#### Custom Chunking Configuration

```python
from koboldapi import ChunkingProcessor, KoboldAPI

# Initialize with custom chunking parameters
processor = ChunkingProcessor(
    api_client=KoboldAPI("http://localhost:5001"),
    max_chunk_length=4096,
    max_total_chunks=1000
)

# Process with metadata
chunks, metadata = processor.chunk_file("large_document.txt")
```

## Error Handling

The library provides custom exceptions for proper error handling:

```python
from koboldapi import KoboldAPIError

try:
    api = KoboldAPI("http://localhost:5001")
    response = api.generate("Test prompt")
except KoboldAPIError as e:
    print(f"API Error: {e}")
```

## Best Practices

1. **Chunking**: Always use the ChunkingProcessor for large documents to ensure proper token management.

2. **Templates**: Use the InstructTemplate system to ensure proper formatting for different models.

3. **Error Handling**: Implement proper error handling using try/except blocks with KoboldAPIError.

4. **Resource Management**: For large files or batch processing, consider using async methods and proper cleanup.

5. **Image Processing**: Set appropriate max_dimension and quality parameters based on your model's requirements.

## Contributing

Contributions are welcome! Please check our GitHub repository for guidelines.

## License

Apache License 2.0
