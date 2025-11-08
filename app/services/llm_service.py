from typing import Optional, Dict, Any, List
from enum import Enum
import logging
from abc import ABC, abstractmethod

# LLM Provider imports
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from app.core.config import LLMProvider

logger = logging.getLogger(__name__)

class BaseLLMService(ABC):
    """Base class for LLM services"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass

class OpenAIService(BaseLLMService):
    """OpenAI GPT service"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        super().__init__(api_key, model_name)
        if not openai:
            raise ImportError("OpenAI package not installed")
        
        self.client = openai.OpenAI(api_key=api_key)
        
    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_tokens', 4000)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class AnthropicService(BaseLLMService):
    """Anthropic Claude service"""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key, model_name)
        if not anthropic:
            raise ImportError("Anthropic package not installed")
            
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get('max_tokens', 4000),
                temperature=kwargs.get('temperature', 0.1),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

class GoogleService(BaseLLMService):
    """Google Gemini service"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        super().__init__(api_key, model_name)
        if not genai:
            raise ImportError("Google GenerativeAI package not installed")
            
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)
        
    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0.1),
                    max_output_tokens=kwargs.get('max_tokens', 4000)
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise

class LLMServiceFactory:
    """Factory for creating LLM services"""
    
    @staticmethod
    def create_service(provider: LLMProvider, api_key: str, model_name: Optional[str] = None) -> BaseLLMService:
        """Create LLM service based on provider"""
        
        if provider == LLMProvider.OPENAI:
            model = model_name or "gpt-4o"
            return OpenAIService(api_key, model)
            
        elif provider == LLMProvider.ANTHROPIC:
            model = model_name or "claude-opus-4-20250514"
            return AnthropicService(api_key, model)
            
        elif provider == LLMProvider.GOOGLE:
            model = model_name or "gemini-2.5-pro"
            return GoogleService(api_key, model)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

# Main LLM service coordinator
class MultiLLMService:
    """Coordinates multiple LLM providers"""
    
    def __init__(self):
        self.services: Dict[LLMProvider, BaseLLMService] = {}
        
    def add_service(self, provider: LLMProvider, api_key: str, model_name: Optional[str] = None):
        """Add an LLM service"""
        service = LLMServiceFactory.create_service(provider, api_key, model_name)
        self.services[provider] = service
        logger.info(f"Added {provider} service with model {service.model_name}")
        
    def generate_with_provider(self, provider: LLMProvider, prompt: str, **kwargs) -> str:
        """Generate response with specific provider"""
        if provider not in self.services:
            raise ValueError(f"Provider {provider} not configured")
            
        return self.services[provider].generate_response(prompt, **kwargs)
        
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of configured providers"""
        return list(self.services.keys())
