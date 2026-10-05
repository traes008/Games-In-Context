"""LLM model factory and wrapper providing unified interface across multiple providers."""

import logging
import os
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

logger = logging.getLogger(__name__)


class LLMWrapper:
    """
    Wrapper that provides a unified interface for LLM calls and tool binding.
    
    This abstraction allows seamless switching between different LLM providers.
    """

    def __init__(self, llm: Any) -> None:
        """
        Initialize the LLM wrapper.
        
        Args:
            llm: A LangChain BaseChatModel instance.
        """
        self._llm = llm

    def call(self, messages: List) -> Any:
        """
        Invoke the LLM with a list of messages.
        
        Args:
            messages: List of LangChain message objects.
            
        Returns:
            The LLM's response message.
        """
        return self._llm.invoke(messages)

    def bind_tools(self, tools: List[Dict[str, Any]]) -> "LLMWrapper":
        """
        Bind tools to the LLM for function calling.
        
        Args:
            tools: List of tool definitions (JSON schema format).
            
        Returns:
            A new LLMWrapper instance with bound tools.
        """
        return LLMWrapper(self._llm.bind_tools(tools))


class LLMFactory:
    """
    Factory for creating LLM instances with unified interface across providers.
    
    Supported providers:
    - ollama: Local Ollama instances
    - openai: OpenAI API
    - anthropic: Anthropic Claude models
    - google: Google Generative AI
    - huggingface: Hugging Face inference endpoints
    """

    @staticmethod
    def get(
        provider: str,
        model_name: str,
        temperature: float = 0.0,
        **kwargs: Any
    ) -> LLMWrapper:
        """
        Factory method to get an LLM instance.
        
        Args:
            provider: Provider name ('ollama', 'openai', 'anthropic', 'google', 'huggingface').
            model_name: Model identifier specific to the provider.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = random).
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            LLMWrapper instance ready for use.
            
        Raises:
            ValueError: If provider is not recognized.
        """
        logger.info(f"Initializing LLM: {provider}/{model_name} (temperature={temperature})")
        
        if provider == "ollama":
            llm = ChatOllama(model=model_name, temperature=temperature, **kwargs)
            return LLMWrapper(llm)

        if provider == "openai":
            llm = ChatOpenAI(model=model_name, temperature=temperature, **kwargs)
            return LLMWrapper(llm)

        if provider == "anthropic":
            llm = ChatAnthropic(model=model_name, temperature=temperature, **kwargs)
            return LLMWrapper(llm)

        if provider == "google":
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, **kwargs)
            return LLMWrapper(llm)

        if provider == "huggingface":
            # For remote inference endpoints (serverless or dedicated)
            hf_llm = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",
                temperature=temperature,
                **kwargs
            )
            llm = ChatHuggingFace(llm=hf_llm)
            return LLMWrapper(llm)

        error_msg = f"Unknown provider: {provider}. Supported: ollama, openai, anthropic, google, huggingface"
        logger.error(error_msg)
        raise ValueError(error_msg)

    