import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from koboldapi import KoboldAPICore, ChunkingProcessor

TASKS = {
    'translate': {
        'chunk_ratio': 0.45,
        'instruction_template': (
            "Translate the text into {language}. "
            "Maintain linguistic flourish and authorial style."
        )
    },
    'summary': {
        'chunk_ratio': 0.8,
        'instruction_template': (
            "Extract the key points and themes from the text "
            "without developing conclusions."
        )
    },
    'correct': {
        'chunk_ratio': 0.45,
        'instruction_template': (
            "Correct any grammar, spelling, and style errors. "
            "Preserve the original meaning and style."
        )
    },
    'distill': {
        'chunk_ratio': 0.8,
        'instruction_template': (
            "Rewrite the text to be concise without losing meaning."
        )
    }
}

class TextProcessor:
    """ Handles text processing """
    
    def __init__(self, core: KoboldAPICore, max_chunk_size):
        """ Initialize with KoboldAPICore instance. """
        self.api = core.api_client
        self.max_context = self.api.get_max_context_length()
        if (self.max_context) > int(max_chunk_size):
            self.max_context = max_chunk_size
        self.template_wrapper = core.template_wrapper
        
    def _get_task_config(self, task: str, language: str = "English") -> dict:
        """ Get configuration for specified task. """
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
            
        config = TASKS[task].copy()
        config['chunk_size'] = int(self.max_context * config['chunk_ratio'])
        config['instruction'] = config['instruction_template'].format(
            language=language
        )
        return config

    async def _process_chunk(self, chunk: str, instruction: str) -> str:
        """ Process a single chunk with streaming output. """
        wrapped = self.template_wrapper.wrap_prompt(
            instruction=instruction,
            content=chunk
        )
        
        result = []
        async for token in self.api.stream_generate(
            wrapped,
            max_length=self.max_context // 2
        ):
            print(token, end='', flush=True)
            result.append(token)
            
        return ''.join(result)

    async def process_text(self, task: str, file_path: Path, 
                         language: str = "English") -> tuple[list[str], dict]:
        """ Process text document with specified task. """
        task_config = self._get_task_config(task, language)
        
        chunker = ChunkingProcessor(self.api, task_config['chunk_size'])
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

async def process_file(api_url: str, input_path: Path, task: str, 
                      language: str = "English", 
                      max_chunk_size: int = 8192):
    """ Process a text file and save results. """
    try:
        core = KoboldAPICore(api_url)
        processor = TextProcessor(core, max_chunk_size)
        results, metadata = await processor.process_text(task, input_path, language)
        
        print(f"\nProcessing complete. Metadata:")
        print(json.dumps(metadata, indent=2))
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="Process text documents with KoboldCPP"
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Input text file'
    )
    parser.add_argument(
        '--task',
        required=True,
        choices=list(TASKS.keys()),
        help='Processing task to perform'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:5001',
        help='KoboldCPP API URL'
    )
    parser.add_argument(
        '--language',
        default='English',
        help='Target language for translation'
    )
    parser.add_argument(
        '--max-chunk-size',
        default=8192,
        type=int,
        help='Largest number of tokens for a chunk'
    )
    
    args = parser.parse_args()
    return asyncio.run(process_file(
        args.api_url,
        args.input,
        args.task,
        args.language,
        args.max_chunk_size
    ))

if __name__ == '__main__':
    exit(main())