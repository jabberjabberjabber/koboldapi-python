import asyncio
import unittest
import sys
import os
from koboldapi.core.api import KoboldAPI

class TestKoboldAPIStreaming(unittest.TestCase):
    def setUp(self):
        # Replace with your actual KoboldCPP API URL
        self.api_url = "http://localhost:5001"
        self.client = KoboldAPI(self.api_url)

    def test_sync_generation(self):
        """
        Test synchronous generation using the wrapper
        """
        print("\n--- Testing Synchronous Generation ---")
        
        try:
            generated_text = self.client.generate_sync(
                "Once upon a time", 
                max_length=50, 
                temperature=0.7
            )
            
            print("\nFull generated text:")
            print(generated_text)
            
            self.assertTrue(len(generated_text.strip()) > 0, "No text was generated")
            
        except Exception as e:
            self.fail(f"Synchronous generation failed: {e}")

    async def _test_async_streaming(self):
        """
        Test asynchronous streaming generation
        """
        print("\n--- Testing Asynchronous Streaming ---")
        
        generated_text = ""
        
        try:
            async for chunk in self.client.stream_generate(
                "In a galaxy far, far away", 
                max_length=50, 
                temperature=0.7
            ):
                print(chunk, end='', flush=True)
                generated_text += chunk
            
            print("\n\nFull generated text:")
            print(generated_text)
            
            assert len(generated_text.strip()) > 0, "No text was generated"
            
        except Exception as e:
            print(f"Async streaming failed: {e}")
            raise

    def test_async_streaming(self):
        """
        Wrapper to run async test in event loop
        """
        asyncio.run(self._test_async_streaming())

    def test_api_availability(self):
        """
        Quick test to verify API is reachable and basic functionality works
        """
        try:
            version = self.client.get_version()
            print("\nAPI Version:", version)
            self.assertIsNotNone(version, "Failed to retrieve API version")

            model = self.client.get_model()
            print("Current Model:", model)
            self.assertIsNotNone(model, "Failed to retrieve current model")

        except Exception as e:
            self.fail(f"API availability check failed: {e}")

if __name__ == '__main__':
    unittest.main()