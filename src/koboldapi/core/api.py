import requests
import json
import random
from typing import Optional, Dict, Any, List, Union, AsyncIterator, Iterator
import asyncio
import sseclient
import aiohttp

class KoboldAPIError(Exception):
    """ Custom exception for Kobold API errors """
    pass
    
class KoboldAPI:
    """ Client for interacting with KoboldCPP API """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None):
        """ Initialize the client
        
            Args:
                api_url: Base URL for the KoboldCPP API (e.g. http://localhost:5001)
                api_password: Optional API password
        """
        self.api_url = api_url.rstrip('/')
        self.genkey = f"KCPP{''.join(str(random.randint(0, 9)) for _ in range(4))}"
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
            
        self.api_endpoints = {
            "tokencount": {
                "path": "/api/extra/tokencount",
                "method": "POST"
            },
            "generate": {
                "path": "/api/v1/generate",
                "method": "POST"
            },
            "check": {
                "path": "/api/extra/generate/check", 
                "method": "POST"
            },
            "abort": {
                "path": "/api/extra/abort",
                "method": "POST"
            },
            "max_context_length": {
                "path": "/api/extra/true_max_context_length",
                "method": "GET"
            },
            "version": {
                "path": "/api/extra/version",
                "method": "GET"
            },
            "model": {
                "path": "/api/v1/model",
                "method": "GET"
            },
            "performance": {
                "path": "/api/extra/perf",
                "method": "GET"
            },
            "tokenize": {
                "path": "/api/extra/tokenize",
                "method": "POST"
            },
            "detokenize": {
                "path": "/api/extra/detokenize",
                "method": "POST"
            },
            "logprobs": {
                "path": "/api/extra/last_logprobs",
                "method": "POST"
            }
        }

    def _call_api(self, endpoint: str, payload: Optional[Dict] = None) -> Any:
        """ Call the Kobold API 
        
            Args:
                endpoint: Name of the API endpoint to call
                payload: Optional dictionary of data to send
                
            Returns:
                API response data
                
            Raises:
                KoboldAPIError: If API call fails
        """
        if endpoint not in self.api_endpoints:
            raise KoboldAPIError(f"Unknown API endpoint: {endpoint}")   
            
        endpoint_info = self.api_endpoints[endpoint]
        url = f"{self.api_url}{endpoint_info['path']}"
        
        try:
            request_method = getattr(requests, endpoint_info['method'].lower())
            response = request_method(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()    
        except requests.RequestException as e:
            raise KoboldAPIError(f"API request failed: {str(e)}")
        except json.JSONDecodeError:
            raise KoboldAPIError("API returned invalid JSON response")

    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7,
                top_p: float = 0.9, top_k: int = 40, rep_pen: float = 1.1,
                rep_pen_range: int = 256, stop_sequences: Optional[List[str]] = None,
                logprobs: bool = False, images: str = [], min_p: float = 0.02) -> str:
        """ Generate text from a prompt with specified parameters
        
            Args:
                prompt: Text prompt to generate from
                max_length: Maximum number of tokens to generate
                temperature: Sampling temperature (higher = more random)
                top_p: Top-p sampling threshold
                top_k: Top-k sampling threshold  
                rep_pen: Repetition penalty
                rep_pen_range: How many tokens back to apply repetition penalty
                stop_sequences: Optional list of strings that will stop generation
                logprobs: Whether to return token logprobs (default False)
                
            Returns:
                Generated text
        """
        payload = {
            "prompt": prompt,
            "genkey": self.genkey,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "rep_pen": rep_pen,
            "rep_pen_range": rep_pen_range,
            "logprobs": logprobs,
            "images": images,
            "min_p": min_p
        }
        if stop_sequences:
            payload["stop_sequence"] = stop_sequences
            
        try:
            result = self._call_api("generate", payload)
            return result["results"][0]["text"]
        except (KeyError, TypeError):
            raise KoboldAPIError("API response missing expected fields")

    def abort_generation(self) -> bool:
        """ Abort the current ongoing generation
        
            Returns:
                True if successfully aborted, False otherwise
        """
        payload = {"genkey": self.genkey}
        try:
            result = self._call_api("abort", payload)
            return result.get("success", False)
        except:
            return False

    def check_generation(self) -> Optional[str]:
        """ Check status of ongoing generation
        
            Returns:
                Currently generated text if available, None otherwise
        """
        payload = {"genkey": self.genkey}
        try:
            result = self._call_api("check", payload)
            return result["results"][0]["text"]
        except:
            return None

    def count_tokens(self, text: str) -> Dict[str, Union[int, List[int]]]:
        """ Count tokens in a text string
        
            Args:
                text: Text to tokenize
                
            Returns:
                Dict containing token count and token IDs
        """
        payload = {"prompt": text}
        result = self._call_api("tokencount", payload)
        return {
            "count": result["value"],
            "tokens": result["ids"]
        }

    def tokenize(self, text: str) -> List[int]:
        """ Convert text to token IDs
        
            Args:
                text: Text to tokenize
                
            Returns:
                List of token IDs
        """
        payload = {"prompt": text}
        result = self._call_api("tokenize", payload)
        return result["ids"]

    def detokenize(self, token_ids: List[int]) -> str:
        """ Convert token IDs back to text
        
            Args:
                token_ids: List of token IDs
                
            Returns:
                Decoded text
        """
        payload = {"ids": token_ids}
        result = self._call_api("detokenize", payload)
        return result["result"]

    def get_last_logprobs(self) -> Dict:
        """ Get token logprobs from the last generation
        
            Returns:
                Dictionary containing logprob information
        """
        payload = {"genkey": self.genkey}
        result = self._call_api("logprobs", payload)
        return result["logprobs"]
        
    def get_version(self) -> Dict[str, str]:
        """ Get KoboldCPP version info
        
            Returns:
                Dictionary with version information
        """
        return self._call_api("version")

    def get_model(self) -> str:
        """ Get current model name
        
            Returns:
                Model name string
        """
        result = self._call_api("model")
        return result["result"]

    def get_performance_stats(self) -> Dict:
        """ Get performance statistics
        
            Returns:
                Dictionary of performance metrics
        """
        return self._call_api("performance")

    def get_max_context_length(self) -> int:
        """ Get maximum allowed context length
        
            Returns:
                Maximum context length in tokens
        """
        result = self._call_api("max_context_length")
        return result["value"]

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """ Generate text with streaming output using SSE
        
            Args:
                prompt: Text prompt to generate from
                **kwargs: Additional generation parameters (same as generate())
                
            Returns:
                AsyncIterator yielding tokens as they are generated
        """
        payload = {
            "prompt": prompt,
            "genkey": self.genkey,
            **{k: v for k, v in kwargs.items() if v is not None}
        }
        url = f"{self.api_url}/api/extra/generate/stream"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                response.raise_for_status()
                
                buffer = ""  # Buffer for incomplete SSE messages
                async for chunk in response.content:
                    buffer += chunk.decode('utf-8')
                    
                    # Process complete SSE messages in buffer
                    while '\n\n' in buffer:
                        message, buffer = buffer.split('\n\n', 1)
                        
                        for line in message.split('\n'):
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    if "token" in data and data["token"]:
                                        yield data["token"]
                                    if data.get("finish_reason") in ["length", "stop"]:
                                        return
                                except json.JSONDecodeError:
                                    continue  # Skip malformed data

    def generate_sync(self, prompt: str, **kwargs) -> str:
        """ Synchronous version of streaming generation that returns complete text
            
            Args:
                prompt: Text prompt to generate from
                **kwargs: Additional generation parameters
                
            Returns:
                Complete generated text as a single string
        """
        result = []
        async def collect():
            async for token in self.stream_generate(prompt, **kwargs):
                result.append(token)
                
        asyncio.run(collect())
        return ''.join(result)