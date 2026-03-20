"""Provider-agnostic VLM clients used by the planner."""

import os
import json
import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from dronebuddylib.utils.logger import Logger

logger = Logger()


class VLMProvider(Enum):
    """Supported VLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class VLMMessage:
    """A message in the conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None


@dataclass
class VLMResponse:
    """Response from a VLM."""
    content: str
    model: str
    provider: VLMProvider
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseVLMClient(ABC):
    """Abstract base class for VLM clients."""
    
    def __init__(self, api_key: str, model: str, temperature: float = 0.3):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.conversation_history: List[VLMMessage] = []
        self.system_prompt: Optional[str] = None
    
    @abstractmethod
    def get_provider(self) -> VLMProvider:
        """Return the provider type."""
        pass
    
    @abstractmethod
    def send_message(self, message: str, image_path: Optional[str] = None) -> VLMResponse:
        """Send a message and get a response."""
        pass
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the conversation."""
        self.system_prompt = prompt
        # Remove any existing system message
        self.conversation_history = [
            msg for msg in self.conversation_history if msg.role != "system"
        ]
    
    def add_message(self, role: str, content: str, image_path: Optional[str] = None):
        """Add a message to conversation history."""
        self.conversation_history.append(VLMMessage(role=role, content=content, image_path=image_path))
    
    def clear_history(self):
        """Clear conversation history (keeps system prompt)."""
        self.conversation_history = []
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_image_media_type(self, image_path: str) -> str:
        """Get the media type from file extension."""
        ext = os.path.splitext(image_path)[1].lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return media_types.get(ext, "image/jpeg")


class OpenAIClient(BaseVLMClient):
    """OpenAI VLM client (GPT-4, GPT-4o, GPT-5, etc.)."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.3):
        super().__init__(api_key, model, temperature)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.log_info('OpenAIClient', f'Initialized with model: {model}')
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def get_provider(self) -> VLMProvider:
        return VLMProvider.OPENAI
    
    def send_message(self, message: str, image_path: Optional[str] = None) -> VLMResponse:
        """Send message to OpenAI API."""
        messages = []
        
        # Add system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        for msg in self.conversation_history:
            if msg.image_path:
                content = [
                    {"type": "text", "text": msg.content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{self._get_image_media_type(msg.image_path)};base64,{self._encode_image(msg.image_path)}"
                        }
                    }
                ]
                messages.append({"role": msg.role, "content": content})
            else:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message
        if image_path:
            content = [
                {"type": "text", "text": message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self._get_image_media_type(image_path)};base64,{self._encode_image(image_path)}"
                    }
                }
            ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": message})
        
        # Call API with error handling
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
        except Exception as e:
            logger.log_error('OpenAIClient', f'API call failed: {e}')
            raise RuntimeError(f"OpenAI API call failed: {e}")
        
        assistant_message = response.choices[0].message.content
        
        # Add to history
        self.add_message("user", message, image_path)
        self.add_message("assistant", assistant_message)
        
        return VLMResponse(
            content=assistant_message,
            model=self.model,
            provider=VLMProvider.OPENAI,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            raw_response=response
        )


class AnthropicClient(BaseVLMClient):
    """Anthropic VLM client (Claude models)."""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.3):
        super().__init__(api_key, model, temperature)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.log_info('AnthropicClient', f'Initialized with model: {model}')
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    def get_provider(self) -> VLMProvider:
        return VLMProvider.ANTHROPIC
    
    def send_message(self, message: str, image_path: Optional[str] = None) -> VLMResponse:
        """Send message to Anthropic API."""
        messages = []
        
        # Anthropic receives the system prompt separately.
        for msg in self.conversation_history:
            if msg.role == "system":
                continue
            
            if msg.image_path:
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": self._get_image_media_type(msg.image_path),
                            "data": self._encode_image(msg.image_path)
                        }
                    },
                    {"type": "text", "text": msg.content}
                ]
                messages.append({"role": msg.role, "content": content})
            else:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message
        if image_path:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self._get_image_media_type(image_path),
                        "data": self._encode_image(image_path)
                    }
                },
                {"type": "text", "text": message}
            ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": message})
        
        # Call API with error handling
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt or "",
                messages=messages,
                temperature=self.temperature
            )
        except Exception as e:
            logger.log_error('AnthropicClient', f'API call failed: {e}')
            raise RuntimeError(f"Anthropic API call failed: {e}")
        
        assistant_message = response.content[0].text
        
        # Add to history
        self.add_message("user", message, image_path)
        self.add_message("assistant", assistant_message)
        
        return VLMResponse(
            content=assistant_message,
            model=self.model,
            provider=VLMProvider.ANTHROPIC,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            } if response.usage else None,
            raw_response=response
        )


class GoogleClient(BaseVLMClient):
    """Google VLM client (Gemini models)."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro", temperature: float = 0.3):
        super().__init__(api_key, model, temperature)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            self.chat = None
            logger.log_info('GoogleClient', f'Initialized with model: {model}')
        except ImportError:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
    
    def get_provider(self) -> VLMProvider:
        return VLMProvider.GOOGLE
    
    def _start_chat(self):
        """Start or restart a chat session."""
        import google.generativeai as genai
        history = []
        
        for msg in self.conversation_history:
            if msg.role == "system":
                continue
            
            role = "user" if msg.role == "user" else "model"
            
            if msg.image_path:
                import PIL.Image
                img = PIL.Image.open(msg.image_path)
                history.append({"role": role, "parts": [img, msg.content]})
            else:
                history.append({"role": role, "parts": [msg.content]})
        
        self.chat = self.client.start_chat(history=history)
    
    def send_message(self, message: str, image_path: Optional[str] = None) -> VLMResponse:
        """Send message to Google Gemini API."""
        import google.generativeai as genai
        
        # Prefix system prompt for provider consistency.
        full_message = message
        if self.system_prompt:
            full_message = f"{self.system_prompt}\n\n---\n\n{message}"
        
        # Start chat if needed
        if self.chat is None:
            self._start_chat()
        
        # Prepare content
        if image_path:
            import PIL.Image
            img = PIL.Image.open(image_path)
            content = [img, full_message]
        else:
            content = full_message
        
        # Send message with error handling
        try:
            response = self.chat.send_message(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature
                )
            )
        except Exception as e:
            logger.log_error('GoogleClient', f'API call failed: {e}')
            raise RuntimeError(f"Google Gemini API call failed: {e}")
        
        assistant_message = response.text
        
        # Add to history
        self.add_message("user", message, image_path)
        self.add_message("assistant", assistant_message)
        
        return VLMResponse(
            content=assistant_message,
            model=self.model,
            provider=VLMProvider.GOOGLE,
            usage=None,  # Gemini usage tracking differs
            raw_response=response
        )


def create_vlm_client(
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    temperature: float = 0.3
) -> BaseVLMClient:
    """
    Factory function to create a VLM client for the specified provider.
    
    Args:
        provider: Provider name ("openai", "anthropic", "google")
        api_key: API key for the provider
        model: Model name (uses provider default if not specified)
        temperature: Temperature for response generation
        
    Returns:
        A VLM client instance
        
    Example:
        client = create_vlm_client("anthropic", "sk-ant-...", "claude-3-5-sonnet-20241022")
        response = client.send_message("Hello!")
    """
    provider_lower = provider.lower()
    
    # Provider defaults.
    default_models = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
        "google": "gemini-1.5-pro"
    }
    
    if provider_lower == "openai":
        return OpenAIClient(api_key, model or default_models["openai"], temperature)
    elif provider_lower == "anthropic":
        return AnthropicClient(api_key, model or default_models["anthropic"], temperature)
    elif provider_lower == "google":
        return GoogleClient(api_key, model or default_models["google"], temperature)
    else:
        raise ValueError(f"Unsupported VLM provider: {provider}. Supported: openai, anthropic, google")
