from typing import List, Dict, Iterator, Tuple, Union
from extractous import Extractor
from koboldapi.chunking.chunker_regex import chunk_regex 

class ChunkingProcessor:
    """ Splits text into chunks appropriate for LLM processing """
    
    def __init__(self, api_client, max_chunk_length: int):
        """ Initialize chunking processor
        
            Args:
                api_client: KoboldAPI instance for token counting
                max_chunk_length: Maximum tokens per chunk
        """
        self.api_client = api_client
        self.max_chunk = max_chunk_length

    def chunk_text(self, content: str) -> List[Tuple[str, int]]:
        """ Split content into appropriate chunks
        
            Args:
                content: Text content to chunk
                
            Returns:
                List of (chunk, token_count) tuples
        """
        chunks = []
        remaining = content
        chunk_num = 0
        
        while remaining:
            # KoboldCPP API has 50k char limit - use 45k for safety
            current_section = remaining[:45000]
            remaining = remaining[45000:]
            
            chunk = self._get_chunk(current_section)
            chunk_len = len(chunk)
            
            if chunk_len == 0:
                print("Warning: Got zero-length chunk")
                continue
                
            chunk_tokens = self.api_client.count_tokens(chunk)["count"]
            chunks.append((chunk, chunk_tokens))
            
            remaining = current_section[len(chunk):].strip() + remaining
            chunk_num += 1
            print(f"Chunked: {chunk_num}")

        return chunks

    def _get_chunk(self, content: str) -> str:
        """ Get appropriate sized chunk using natural breaks
        
            Args:
                content: Text content to chunk
                
            Returns:
                Chunk of text
        """
        total_tokens = self.api_client.count_tokens(content)["count"]
        if total_tokens < self.max_chunk:
            return content

        # chunk_regex is designed to break at natural language points
        # to preserve context and readability
        matches = chunk_regex.finditer(content)
        current_size = 0
        chunks = []
        
        for match in matches:
            chunk = match.group(0)
            chunk_size = self.api_client.count_tokens(chunk)["count"]
            if current_size + chunk_size > self.max_chunk:
                if not chunks:
                    chunks.append(chunk)
                break
            chunks.append(chunk)
            current_size += chunk_size
        
        return ''.join(chunks)

    def chunk_file(self, file_path) -> Tuple[List[Tuple[str, int]], Dict]:
        """ Chunk text from file
        
            Args:
                file_path: Path to text file (str or Path object)
                
            Returns:
                Tuple of (chunks with token counts, file metadata)
        """
        # Convert to string for extractous
        file_path_str = str(file_path)
        
        extractor = Extractor()
        extractor = extractor.set_extract_string_max_length(100000000)
        content, metadata = extractor.extract_file_to_string(file_path_str)
        return self.chunk_text(content), metadata