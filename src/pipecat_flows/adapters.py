#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List

from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.google import GoogleLLMService
from pipecat.services.openai import OpenAILLMService


class LLMProvider(Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI = "openai"


class LLMAdapter(ABC):
    """Base adapter for LLM-specific format handling."""

    @abstractmethod
    def get_function_name(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from provider-specific function definition."""
        pass

    @abstractmethod
    def get_function_args(self, function_call: Dict[str, Any]) -> dict:
        """Extract function arguments from provider-specific function call."""
        pass

    @abstractmethod
    def get_message_content(self, message: Dict[str, Any]) -> str:
        """Extract message content from provider-specific format."""
        pass

    @abstractmethod
    def format_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format functions for provider-specific use."""
        pass


class OpenAIAdapter(LLMAdapter):
    """Format adapter for OpenAI."""

    def get_function_name(self, function_def: Dict[str, Any]) -> str:
        return function_def["function"]["name"]

    def get_function_args(self, function_call: Dict[str, Any]) -> dict:
        return function_call.get("arguments", {})

    def get_message_content(self, message: Dict[str, Any]) -> str:
        return message["content"]

    def format_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return functions  # OpenAI format is our default


class AnthropicAdapter(LLMAdapter):
    """Format adapter for Anthropic."""

    def get_function_name(self, function_def: Dict[str, Any]) -> str:
        return function_def["name"]

    def get_function_args(self, function_call: Dict[str, Any]) -> dict:
        return function_call.get("arguments", {})

    def get_message_content(self, message: Dict[str, Any]) -> str:
        if isinstance(message.get("content"), list):
            return " ".join(item["text"] for item in message["content"] if item["type"] == "text")
        return message.get("content", "")

    def format_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for func in functions:
            if "function" in func:
                # Convert from OpenAI format
                formatted.append(
                    {
                        "name": func["function"]["name"],
                        "description": func["function"].get("description", ""),
                        "input_schema": func["function"].get("parameters", {}),
                    }
                )
            else:
                # Already in Anthropic format
                formatted.append(func)
        return formatted


class GeminiAdapter(LLMAdapter):
    """Format adapter for Google's Gemini."""

    def get_function_name(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from provider-specific function definition."""
        if "name" in function_def:
            return function_def["name"]
        return ""

    def get_function_args(self, function_call: Dict[str, Any]) -> dict:
        """Extract function arguments from provider-specific function call."""
        return function_call.get("args", {})

    def get_message_content(self, message: Dict[str, Any]) -> str:
        """Extract message content from provider-specific format."""
        return message["content"]

    def format_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format functions for provider-specific use."""
        all_declarations = []
        for func in functions:
            if "function_declarations" in func:
                all_declarations.extend(func["function_declarations"])
            elif "function" in func:
                all_declarations.append(
                    {
                        "name": func["function"]["name"],
                        "description": func["function"].get("description", ""),
                        "parameters": func["function"].get("parameters", {}),
                    }
                )
            else:
                all_declarations.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )
        return [{"function_declarations": all_declarations}] if all_declarations else []


def create_adapter(llm) -> LLMAdapter:
    """Create appropriate adapter based on LLM service type.

    Args:
        llm: LLM service instance

    Returns:
        LLMAdapter: Provider-specific adapter

    Raises:
        ValueError: If LLM type is not supported
    """
    if isinstance(llm, OpenAILLMService):
        return OpenAIAdapter()
    elif isinstance(llm, AnthropicLLMService):
        return AnthropicAdapter()
    elif isinstance(llm, GoogleLLMService):
        return GeminiAdapter()
    raise ValueError(f"Unsupported LLM type: {type(llm)}")
