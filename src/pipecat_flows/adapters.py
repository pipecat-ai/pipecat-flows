#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM provider adapters for normalizing function and message formats.

This module provides adapters that normalize interactions between different
LLM providers (OpenAI, Anthropic, Gemini). It handles:
- Function name extraction
- Argument parsing
- Message content formatting
- Provider-specific schema conversion

The adapter system allows the flow manager to work with different LLM
providers while maintaining a consistent internal format (based on OpenAI's
function calling convention).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger


class LLMAdapter(ABC):
    """Base adapter for LLM-specific format handling.

    Adapters normalize differences between LLM providers:
    - OpenAI: Uses function calling format
    - Anthropic: Uses native function format
    - Google: Uses function declarations format

    This allows the flow system to work consistently across
    different LLM providers while handling format differences
    internally.
    """

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

    @abstractmethod
    def get_recent_tool_call_pairs(self, messages: List[dict]) -> List[dict]:
        """Get recent consecutive tool calls and results from message history.

        Collects all consecutive tool call pairs working backwards from the end
        of the message history until reaching a regular message interaction.

        Args:
            messages: List of messages in provider's format

        Returns:
            List[dict]: List of messages containing tool calls and results in
                       chronological order, empty list if none found
        """
        pass

    def format_summary_message(self, summary: str) -> dict:
        """Format a summary as a message appropriate for this LLM provider.

        Args:
            summary: The generated summary text

        Returns:
            A properly formatted message for this provider
        """
        pass

    async def generate_summary(
        self, llm: Any, summary_prompt: str, messages: List[dict]
    ) -> Optional[str]:
        """Generate a summary using the LLM provider's API directly.

        Args:
            llm: LLM service instance containing client/credentials
            summary_prompt: Prompt text to guide summary generation
            messages: List of messages to summarize

        Returns:
            Generated summary text, or None if generation fails
        """
        pass


class OpenAIAdapter(LLMAdapter):
    """Format adapter for OpenAI.

    Handles OpenAI's function calling format, which is used as the default format
    in the flow system.
    """

    def get_function_name(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from OpenAI function definition.

        Args:
            function_def: OpenAI-formatted function definition dictionary

        Returns:
            Function name from the definition
        """
        return function_def["function"]["name"]

    def get_function_args(self, function_call: Dict[str, Any]) -> dict:
        """Extract arguments from OpenAI function call.

        Args:
            function_call: OpenAI-formatted function call dictionary

        Returns:
            Dictionary of function arguments, empty if none provided
        """
        return function_call.get("arguments", {})

    def get_message_content(self, message: Dict[str, Any]) -> str:
        """Extract content from OpenAI message format.

        Args:
            message: OpenAI-formatted message dictionary

        Returns:
            Message content as string
        """
        return message["content"]

    def format_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format functions for OpenAI use.

        Args:
            functions: List of function definitions

        Returns:
            Functions in OpenAI format (unchanged as this is our default format)
        """
        return functions

    def get_recent_tool_call_pairs(self, messages: List[dict]) -> List[dict]:
        """Gets consecutive function call/response pairs from message history.

        Processes messages in reverse order to find the most recent function
        interactions. Stops at the first regular message to maintain
        conversation context.

        Args:
        messages: List of messages in OpenAI format

        Returns:
        List of messages containing matched function call/response pairs
        in chronological order.

        Note:
        Function calls must be from "assistant" role with tool_calls,
        and responses must have "tool" role with matching tool_call_id.
        """
        tool_messages = []

        for i in range(len(messages) - 1, -1, -1):
            # If we hit a regular message, stop collecting
            if (
                messages[i].get("role") in ["user", "assistant"]
                and not messages[i].get("tool_calls")
                and not messages[i].get("role") == "tool"
            ):
                break

            # Collect tool call pairs
            if (
                messages[i].get("role") == "tool"
                and i > 0
                and messages[i - 1].get("role") == "assistant"
                and messages[i - 1].get("tool_calls")
            ):
                # Insert at beginning to maintain chronological order
                tool_messages[0:0] = [messages[i - 1], messages[i]]

        return tool_messages

    def format_summary_message(self, summary: str) -> dict:
        """Format summary as a system message for OpenAI."""
        return {"role": "system", "content": f"Here's a summary of the conversation:\n{summary}"}

    async def generate_summary(
        self, llm: Any, summary_prompt: str, messages: List[dict]
    ) -> Optional[str]:
        """Generate summary using OpenAI's API directly."""
        try:
            prompt_messages = [
                {
                    "role": "system",
                    "content": summary_prompt,
                },
                {
                    "role": "user",
                    "content": f"Conversation history: {messages}",
                },
            ]

            # LLM completion
            response = await llm._client.chat.completions.create(
                model=llm.model_name,
                messages=prompt_messages,
                stream=False,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI summary generation failed: {e}", exc_info=True)
            return None


class AnthropicAdapter(LLMAdapter):
    """Format adapter for Anthropic.

    Handles Anthropic's native function format, converting between OpenAI's format
    and Anthropic's as needed.
    """

    def get_function_name(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from Anthropic function definition.

        Args:
            function_def: Anthropic-formatted function definition dictionary

        Returns:
            Function name from the definition
        """
        return function_def["name"]

    def get_function_args(self, function_call: Dict[str, Any]) -> dict:
        """Extract arguments from Anthropic function call.

        Args:
            function_call: Anthropic-formatted function call dictionary

        Returns:
            Dictionary of function arguments, empty if none provided
        """
        return function_call.get("arguments", {})

    def get_message_content(self, message: Dict[str, Any]) -> str:
        """Extract content from Anthropic message format.

        Handles both string content and structured content arrays.

        Args:
            message: Anthropic-formatted message dictionary

        Returns:
            Message content as string, concatenated if from multiple parts
        """
        if isinstance(message.get("content"), list):
            return " ".join(item["text"] for item in message["content"] if item["type"] == "text")
        return message.get("content", "")

    def format_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format functions for Anthropic use.

        Converts from OpenAI format to Anthropic's native function format if needed.

        Args:
            functions: List of function definitions in OpenAI format

        Returns:
            Functions converted to Anthropic's format
        """
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

    def get_recent_tool_call_pairs(self, messages: List[dict]) -> List[dict]:
        """Gets consecutive function call/response pairs from message history.

        Processes messages in reverse order to find the most recent function
        interactions. Stops at the first regular message to maintain
        conversation context.

        Args:
        messages: List of messages in Anthropic format

        Returns:
        List of messages containing matched function call/response pairs
        in chronological order.

        Note:
        Function calls must be from "assistant" role with tool_use content,
        and responses must be from "user" role with tool_result content.
        """
        tool_messages = []

        for i in range(len(messages) - 1, -1, -1):
            content = messages[i].get("content", [])
            # Handle both string and list content formats
            content_list = (
                content if isinstance(content, list) else [{"type": "text", "text": content}]
            )

            # If we hit a regular message, stop collecting
            if all(item.get("type", "text") == "text" for item in content_list):
                break

            # Collect tool call pairs
            if (
                messages[i].get("role") == "user"
                and any(item.get("type") == "tool_result" for item in content_list)
                and i > 0
                and messages[i - 1].get("role") == "assistant"
                and isinstance(messages[i - 1].get("content"), list)
                and any(
                    item.get("type") == "tool_use" for item in messages[i - 1].get("content", [])
                )
            ):
                tool_messages[0:0] = [messages[i - 1], messages[i]]

        return tool_messages

    def format_summary_message(self, summary: str) -> dict:
        """Format summary as a user message for Anthropic."""
        return {"role": "user", "content": f"Here's a summary of the conversation:\n{summary}"}

    async def generate_summary(
        self, llm: Any, summary_prompt: str, messages: List[dict]
    ) -> Optional[str]:
        """Generate summary using Anthropic's API directly."""
        try:
            prompt_messages = [
                {
                    "role": "user",
                    "content": f"Conversation history: {messages}",
                },
            ]

            # LLM completion
            response = await llm._client.messages.create(
                model=llm.model_name,
                messages=prompt_messages,
                system=summary_prompt,
                max_tokens=8192,
                stream=False,
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic summary generation failed: {e}", exc_info=True)
            return None


class GeminiAdapter(LLMAdapter):
    """Format adapter for Google's Gemini.

    Handles Gemini's function declarations format, converting between OpenAI's format
    and Gemini's as needed.
    """

    def get_function_name(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from Gemini function definition.

        Args:
            function_def: Gemini-formatted function definition dictionary

        Returns:
            Function name from the first declaration, or empty string if none found
        """
        logger.debug(f"Getting function name from: {function_def}")
        if "function_declarations" in function_def:
            declarations = function_def["function_declarations"]
            if declarations and isinstance(declarations, list):
                return declarations[0]["name"]
        return ""

    def get_function_args(self, function_call: Dict[str, Any]) -> dict:
        """Extract arguments from Gemini function call.

        Args:
            function_call: Gemini-formatted function call dictionary

        Returns:
            Dictionary of function arguments, empty if none provided
        """
        return function_call.get("args", {})

    def get_message_content(self, message: Dict[str, Any]) -> str:
        """Extract content from Gemini message format.

        Args:
            message: Gemini-formatted message dictionary

        Returns:
            Message content as string
        """
        return message["content"]

    def format_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format functions for Gemini use.

        Converts from OpenAI format to Gemini's function declarations format.

        Args:
            functions: List of function definitions in OpenAI format

        Returns:
            Functions converted to Gemini's format with declarations wrapper
        """
        all_declarations = []
        for func in functions:
            if "function_declarations" in func:
                # Process each declaration separately
                for decl in func["function_declarations"]:
                    formatted_decl = {
                        "name": decl["name"],
                        "description": decl.get("description", ""),
                        "parameters": decl.get("parameters", {"type": "object", "properties": {}}),
                    }
                    all_declarations.append(formatted_decl)
            elif "function" in func:
                all_declarations.append(
                    {
                        "name": func["function"]["name"],
                        "description": func["function"].get("description", ""),
                        "parameters": func["function"].get("parameters", {}),
                    }
                )
        return [{"function_declarations": all_declarations}] if all_declarations else []

    def get_recent_tool_call_pairs(self, messages: List[dict]) -> List[dict]:
        """Gets consecutive function call/response pairs from message history.

        Processes messages in reverse order to find the most recent function
        interactions. Stops at the first regular text message to maintain
        conversation context.

        Args:
            messages: List of messages in Gemini format

        Returns:
            List of messages containing matched function call/response pairs
            in chronological order.

        Note:
            Function calls must be from "model" role and responses from "user" role.
            Names must match between call and response.
        """
        tool_messages = []

        # Process messages in reverse order
        for i in range(len(messages) - 1, -1, -1):
            current_message = messages[i]

            # Stop at first regular text message
            if len(current_message.parts) > 0 and str(current_message.parts[0]).startswith("text:"):
                break

            try:
                part = current_message.parts[0]
                # Check if current message is a function response
                is_function_response = (
                    current_message.role == "user"
                    and len(current_message.parts) > 0
                    and str(part).startswith("function_response {")
                )

                if is_function_response and i > 0:
                    prev_message = messages[i - 1]
                    prev_part = prev_message.parts[0]

                    # Check if previous message is matching function call
                    is_matching_function_call = (
                        prev_message.role == "model"
                        and len(prev_message.parts) > 0
                        and str(prev_part).startswith("function_call {")
                        and prev_part.function_call.name == part.function_response.name
                    )

                    # Add pair to start of list to maintain chronological order
                    if is_matching_function_call:
                        tool_messages[0:0] = [prev_message, current_message]

            except Exception:
                continue

        return tool_messages

    def format_summary_message(self, summary: str) -> dict:
        """Format summary as a user message for Gemini."""
        return {"role": "user", "content": f"Here's a summary of the conversation:\n{summary}"}

    async def generate_summary(
        self, llm: Any, summary_prompt: str, messages: List[dict]
    ) -> Optional[str]:
        """Generate summary using Google's API directly."""
        try:
            # Format messages for Gemini
            contents = [
                {
                    "role": "user",
                    "parts": [{"text": (f"{summary_prompt}\n\nConversation history: {messages}")}],
                }
            ]

            # Use non-streaming completion
            response = await llm._client.generate_content_async(contents=contents, stream=False)

            return response.text

        except Exception as e:
            logger.error(f"Google summary generation failed: {e}", exc_info=True)
            return None


def create_adapter(llm) -> LLMAdapter:
    """Create appropriate adapter based on LLM service type.

    Uses lazy imports to avoid requiring all provider dependencies at runtime.
    Only the dependency for the chosen provider needs to be installed.

    Args:
        llm: LLM service instance

    Returns:
        LLMAdapter: Provider-specific adapter

    Raises:
        ValueError: If LLM type is not supported or required dependency not installed
    """
    # Try OpenAI
    try:
        from pipecat.services.openai import OpenAILLMService

        if isinstance(llm, OpenAILLMService):
            logger.debug("Creating OpenAI adapter")
            return OpenAIAdapter()
    except ImportError as e:
        logger.debug(f"OpenAI import failed: {e}")

    # Try Anthropic
    try:
        from pipecat.services.anthropic import AnthropicLLMService

        if isinstance(llm, AnthropicLLMService):
            logger.debug("Creating Anthropic adapter")
            return AnthropicAdapter()
    except ImportError as e:
        logger.debug(f"Anthropic import failed: {e}")

    # Try Google
    try:
        from pipecat.services.google import GoogleLLMService
        from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService

        if isinstance(llm, GoogleLLMService) or isinstance(llm, GeminiMultimodalLiveLLMService):
            logger.debug("Creating Google adapter")
            return GeminiAdapter()
    except ImportError as e:
        logger.debug(f"Google import failed: {e}")

    # If we get here, either the LLM type is not supported or the required dependency is not installed
    llm_type = type(llm).__name__
    error_msg = f"Unsupported LLM type or missing dependency: {llm_type}\n"
    error_msg += "Make sure you have installed the required dependency:\n"
    error_msg += "- For OpenAI: pip install 'pipecat-ai[openai]'\n"
    error_msg += "- For Anthropic: pip install 'pipecat-ai[anthropic]'\n"
    error_msg += "- For Google: pip install 'pipecat-ai[google]'"

    raise ValueError(error_msg)
