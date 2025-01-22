from typing import Optional, Dict, Any, List, Union, AsyncIterator
import requests
import json
import random
import asyncio
import aiohttp

class KoboldAPIError(Exception):
    """ Custom exception for Kobold API errors """
    pass
    
class KoboldAPI:
    def __init__(self, api_url: str, api_password: Optional[str] = None,
                 generation_params: Optional[Dict] = None,
                 templates_directory: Optional[str] = None):
        self.api_url = api_url.rstrip('/')
        self.api_password = api_password
        self.templates_directory = templates_directory
        self.generation_params = {
            'temp': 0,
            'top_k': 0,
            'top_p': 1.0,
            'rep_pen': 1.0,
            'min_p': 0.05
        }
        if generation_params:
            self.generation_params.update(generation_params)
            
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
        
    def _call_api(self, endpoint: str, payload: Optional[Dict] = None):
        """ Call the Kobold API with proper error handling """
        if endpoint not in self.api_endpoints:
            raise KoboldAPIError(f"Unknown API endpoint: {endpoint}")   
            
        endpoint_info = self.api_endpoints[endpoint]
        url = f"{self.api_url}{endpoint_info['path']}"
        
        try:
            request_method = getattr(requests, endpoint_info['method'].lower())
            response = request_method(
                url, 
                json=payload, 
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()    
        except requests.RequestException as e:
            raise KoboldAPIError(f"API request failed: {str(e)}")
        except json.JSONDecodeError:
            raise KoboldAPIError("API returned invalid JSON response")

    def generate(self, prompt: str, max_length: int = 300,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                **kwargs) -> str:
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "genkey": self.genkey,
            **kwargs
        }
        result = self._call_api("generate", payload)
        if not result.get("results"):
            raise KoboldAPIError("API response missing results")
        return result["results"][0]["text"]
        
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
                text: Text to count tokens from
                
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
    
    def update_generation_params(self, new_params: Dict):
        """ Update generation parameters """
        self.generation_params.update(new_params)
