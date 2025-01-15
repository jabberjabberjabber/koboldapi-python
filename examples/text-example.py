import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from koboldapi import (
            KoboldAPICore, 
            ChunkingProcessor, 
            KoboldAPIConfig)

class TextProcessor:
    """ Handles text processing tasks using chunking and streaming. """
    TASKS = {
        'translate': {
            'chunk_ratio': 0.45,
            'instruction_template': (
                "Translate the text into {language}. "
                "Maintain linguistic flourish and authorial style as much as possible. "
                "Write the full contents without condensing or modernizing."
            )
        },
        'summary': {
            'chunk_ratio': 0.8,
            'instruction_template': (
                "Extract the key points, themes and actions from the text succinctly "
                "without developing any conclusions or commentary."
            )
        },
        'correct': {
            'chunk_ratio': 0.45,
            'instruction_template': (
                "Correct any grammar, spelling, style, or format errors in the text. "
                "Do not alter the text or otherwise change the meaning or style."
            )
        },
        'distill': {
            'chunk_ratio': 0.8,
            'instruction_template': (
                "Rewrite the text to be as concise as possible without losing meaning."
            )
        }
    }

    def __init__(self, core: KoboldAPICore):
        """ Initialize with KoboldAPICore instance. """
        self.core = core
        self.max_context = core.api_client.get_max_context_length()
        
    def _get_task_config(self, task: str, language: str = "English") -> dict:
        """ Get configuration for specified task. """
        if task not in self.TASKS:
            raise ValueError(f"Unknown task: {task}")
            
        config = self.TASKS[task].copy()
        config['chunk_size'] = int(self.max_context * config['chunk_ratio'])
        config['instruction'] = config['instruction_template'].format(
            language=language
        )
        return config

    async def _process_chunk(self, chunk: str, instruction: str) -> str:
        """ Process with streaming output. """
        wrapped = self.core.template_wrapper.wrap_prompt(
            instruction=instruction,
            content=chunk,
            system_instruction="You are a helpful assistant."
        )
        
        generated_text = ""
        async for token in self.core.api_client.stream_generate(
            wrapped,
            max_length=self.max_context // 2,
            temp=0,
            **self.core.get_generation_params()
        ):
            print(token, end='', flush=True)
            generated_text += token
            
        return generated_text

    async def process_text(self, 
                          task: str,
                          file_path: Path,
                          language: str = "English") -> Tuple[List[str], Dict]:
        """ Process text document with specified task.
        
            Args:
                task: Task type ('translate', 'summary', 'correct', 'distill')
                file_path: Path to text file
                language: Target language for translation task
                
            Returns:
                Tuple of (processed chunks, metadata)
        """
        task_config = self._get_task_config(task, language)
        
        chunker = ChunkingProcessor(
            self.core.api_client,
            max_chunk_length=task_config['chunk_size']
        )
        
        chunks, metadata = chunker.chunk_file(file_path)
        print(f"\nProcessing {len(chunks)} chunks...")
        
        results = []
        for i, (chunk, _) in enumerate(chunks, 1):
            print(f"\nChunk {i}/{len(chunks)}:")
            try:
                result = await self._process_chunk(chunk, task_config['instruction'])
                results.append(result)
            except Exception as e:
                print(f"\nError processing chunk {i}: {e}")
                results.append(f"[Error processing chunk {i}]")
                
        metadata.update({
            'processing_time': datetime.now().isoformat(),
            'task': task,
            'chunks_processed': len(chunks),
            'language': language
        })
        
        return results, metadata


async def process_file(config: dict,
                      input_path: Path,
                      task: str,
                      output_path: Path,
                      language: str = "English",
                      metadata_path: Optional[Path] = None) -> int:
    """ Process a text file and save results.
    
        Returns exit code (0 for success, 1 for errors)
    """
    try:
        core = KoboldAPICore(config)
        processor = TextProcessor(core)
        
        results, metadata = await processor.process_text(task, input_path, language)

        output_path.write_text("\n\n".join(results), encoding='utf-8')
        print(f"\nOutput written to {output_path}")

        if metadata_path:
            metadata_path.write_text(
                json.dumps(metadata, indent=2),
                encoding='utf-8'
            )
            print(f"Metadata written to {metadata_path}")
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main():
    """ Main entry point with argument parsing. """
    parser = argparse.ArgumentParser(
        description="Process text documents with KoboldCPP"
    )
    
    parser.add_argument('input',
                       help='Input text file')
    parser.add_argument('--task',
                       required=True,
                       choices=list(TextProcessor.TASKS.keys()),
                       help='Processing task to perform')
    parser.add_argument('--output',
                       required=True,
                       help='Output file path')
    parser.add_argument('--api-url',
                       default='http://localhost:5001',
                       help='KoboldCPP API URL')
    parser.add_argument('--templates',
                       default='./templates',
                       help='Templates directory path')
    parser.add_argument('--language',
                       default='English',
                       help='Target language for translation')
    parser.add_argument('--metadata',
                       help='Save metadata to JSON file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else None
    
    config_dict = {
        "api_url": args.api_url,
        "api_password": "",
        "templates_directory": args.templates,
    }
    
    return asyncio.run(process_file(
        config_dict,
        input_path,
        args.task,
        output_path,
        args.language,
        metadata_path
    ))


if __name__ == '__main__':
    exit(main())
