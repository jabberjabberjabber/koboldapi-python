import argparse
import sys
from pathlib import Path
from datetime import datetime
import asyncio
import json
from typing import List, Dict, Tuple, Optional, Union

from koboldapi import KoboldAPIConfig, KoboldAPICore
from koboldapi.chunking import ChunkingProcessor

def process_text(core: KoboldAPICore, 
                task: str, 
                file_path: Union[str, Path]) -> Tuple[List[str], Dict]:
    """Process text document with specified task.
    
    Args:
        core: KoboldAPICore instance
        task: Task type ('translate', 'summary', 'correct', 'distill')
        file_path: Path to text file
        
    Returns:
        Tuple of (processed chunks, metadata)
    """
    # Configure task parameters
    max_context = core.api_client.get_max_context_length()
    task_configs = {
        'translate': {
            'chunk_size': int(max_context * 0.4),
            'instruction': (
                f"Translate the text into {core.config.translation_language}. "
                "Maintain linguistic flourish and authorial style as much as possible. "
                "Write the full contents without condensing or modernizing."
            )
        },
        'summary': {
            'chunk_size': int(max_context * 0.8),
            'instruction': (
                "Extract the key points, themes and actions from the text succinctly "
                "without developing any conclusions or commentary."
            )
        },
        'correct': {
            'chunk_size': int(max_context * 0.4),
            'instruction': (
                "Correct any grammar, spelling, style, or format errors in the text. "
                "Do not alter the text or otherwise change the meaning or style."
            )
        },
        'distill': {
            'chunk_size': int(max_context * 0.8),
            'instruction': (
                "Rewrite the text to be as concise as possible without losing meaning."
            )
        }
    }

    # Initialize chunker
    chunker = ChunkingProcessor(
        core.api_client,
        max_chunk_length=task_configs[task]['chunk_size']
    )
    
    # Process text in chunks
    chunks, metadata = chunker.chunk_file(file_path)
    results = []
    
    print(f"\nProcessing {len(chunks)} chunks...")
    
    for i, (chunk, _) in enumerate(chunks, 1):
        print(f"\nChunk {i}/{len(chunks)}:")
        
        # Wrap prompt with template
        wrapped = core.template_wrapper.wrap_prompt(
            instruction=task_configs[task]['instruction'],
            content=chunk,
            system_instruction="You are a helpful assistant."
        )
        
        # Generate response with streaming
        try:
            generated_text = ""
            async def stream():
                nonlocal generated_text
                async for token in core.api_client.stream_generate(
                    prompt=wrapped[0],
                    max_length=max_context // 2,
                    **core.get_generation_params()
                ):
                    print(token, end='', flush=True)
                    generated_text += token
                    
            asyncio.run(stream())
            results.append(generated_text)
            
        except Exception as e:
            print(f"\nError processing chunk {i}: {e}", file=sys.stderr)
            results.append(f"[Error processing chunk {i}]")
    
    # Add metadata
    metadata.update({
        'Processing-Time': datetime.now().isoformat(),
        'Task': task,
        'Chunks-Processed': len(chunks)
    })
    
    return results, metadata

def main():
    parser = argparse.ArgumentParser(
        description="Process text documents with KoboldCPP"
    )
    
    parser.add_argument('input', help='Input text file')
    parser.add_argument(
        '--task', 
        required=True,
        choices=['translate', 'summary', 'correct', 'distill'],
        help='Processing task to perform'
    )
    parser.add_argument('--output', required=True,
                       help='Output file path')
    
    # Optional configuration
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--api-url', default='http://localhost:5001',
                       help='KoboldCPP API URL')
    parser.add_argument('--templates', default='templates',
                       help='Templates directory path')
    parser.add_argument('--language', default='English',
                       help='Target language for translation')
    parser.add_argument('--metadata', help='Save metadata to JSON file')
    
    args = parser.parse_args()
    
    try:
        # Set up configuration
        if args.config:
            config = KoboldAPIConfig.from_json(args.config)
        else:
            config = KoboldAPIConfig(
                api_url=args.api_url,
                api_password="",
                templates_directory=args.templates,
                translation_language=args.language
            )
        
        # Initialize core and process file
        core = KoboldAPICore(config)
        results, metadata = process_text(core, args.task, args.input)
        
        # Save results
        output_path = Path(args.output)
        output_path.write_text(
            "\n\n".join(results),
            encoding='utf-8'
        )
        print(f"\nOutput written to {output_path}")
        
        # Save metadata if requested
        if args.metadata:
            metadata_path = Path(args.metadata)
            metadata_path.write_text(
                json.dumps(metadata, indent=2),
                encoding='utf-8'
            )
            print(f"Metadata written to {metadata_path}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    return 0

if __name__ == '__main__':
    exit(main())