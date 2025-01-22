from typing import List, Dict, Iterator, Tuple
from extractous import Extractor
from koboldapi.chunking.chunker_regex import chunk_regex

class ChunkingProcessor:
    def __init__(self, api_client: "KoboldAPI", 
                 max_chunk_length: int,
                 max_total_chunks: int = 1000):
        if max_chunk_length <= 0:
            raise ValueError("max_chunk_length must be positive")
        self.api_client = api_client
        self.max_chunk = max_chunk_length
        self.max_total_chunks = max_total_chunks

    def chunk_text(self, content: str) -> List[Tuple[str, int]]:
        """ Split content into chunks """
        if not content:
            return []
            
        chunks = []
        remaining = content
        chunk_num = 0
        
        while remaining and chunk_num < self.max_total_chunks:
            #koboldcpp has max char limit of 50k
            current_section = remaining[:45000]
            remaining = remaining[45000:]
            
            chunk = self._get_chunk(current_section)
            chunk_len = len(chunk)
            
            if chunk_len == 0:
                continue
                
            chunk_tokens = self.api_client.count_tokens(chunk)["count"]
            chunks.append((chunk, chunk_tokens))
            remaining = current_section[len(chunk):].strip() + remaining

            chunk_num += 1
            
        if remaining and chunk_num >= self.max_total_chunks:
            raise ValueError(f"Text exceeded maximum of {self.max_total_chunks} chunks")
            
        return chunks

    def _get_chunk(self, content: str) -> str:
        """ Get appropriate sized chunk using natural breaks
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

        extractor = Extractor()
        extractor = extractor.set_extract_string_max_length(100000000)
        content, metadata = extractor.extract_file_to_string(str(file_path))
        return self.chunk_text(content), metadata