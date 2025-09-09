#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM provider adapters for normalizing function and message formats.

This module provides adapters that normalize interactions between different
LLM providers (OpenAI, Anthropic, Gemini, AWS Bedrock). It handles:

- Function name extraction
- Argument parsing
- Message content formatting
- Provider-specific schema conversion

The adapter system allows the flow manager to work with different LLM
providers while maintaining a consistent internal format.
"""

import sys
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext, NotGiven
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from pipecat_flows.types import FlowsDirectFunctionWrapper, FlowsFunctionSchema


class LLMAdapter:
    """Base adapter for LLM-specific format handling.

    Adapters normalize differences between LLM providers to allow the flow system
    to work consistently across different providers while handling format differences
    internally. Each provider has specific requirements for function calling,
    message formatting, and tool definitions.

    Supported providers:

    - OpenAI: Uses function calling format
    - Anthropic: Uses native function format
    - Google Gemini: Uses function declarations format
    - AWS Bedrock: Uses Anthropic-compatible format
    """

    def get_function_name(self, function_def: Union[Dict[str, Any], FlowsFunctionSchema]) -> str:
        """Extract function name from provider-specific function definition or schema.

        Args:
            function_def: Provider-specific function definition or schema.

        Returns:
            Function name extracted from the definition.
        """
        if isinstance(function_def, (FlowsFunctionSchema)):
            return function_def.name
        return self._get_function_name_from_dict(function_def)

    def _get_function_name_from_dict(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from provider-specific function definition.

        Args:
            function_def: Provider-specific function definition dictionary.

        Returns:
            Function name extracted from the definition.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def format_functions(
        self,
        functions: List[Union[Dict[str, Any], FunctionSchema, FlowsFunctionSchema]],
        original_configs: Optional[List] = None,
    ) -> List[Dict[str, Any]] | ToolsSchema | NotGiven:
        """Format functions for provider-specific use.

        The formatted functions/tools will be used in an LLMSetToolsFrame.

        The NotGiven return type is only used by the subclass
        UniversalLLMAdapter, as the universal LLMContext expects NOT_GIVEN when
        there are no tools.

        All other LLM-specific contexts expect either a list of dicts or a
        ToolsSchema, with an empty list meaning "no tools".

        Args:
            functions: List of function definitions (dicts or schema objects).
            original_configs: Optional original node configs, used by some adapters.

        Returns:
            List of functions formatted for the provider.
        """
        # Return empty list to mean "no tools"
        if not functions:
            return []

        # Convert to standard FunctionSchema objects for the ToolsSchema
        standard_functions = []

        for func in functions:
            if isinstance(func, FlowsFunctionSchema):
                # Extract just the FunctionSchema part for the LLM
                standard_functions.append(
                    FunctionSchema(
                        name=func.name,
                        description=func.description,
                        properties=func.properties,
                        required=func.required,
                    )
                )
            elif isinstance(func, FunctionSchema):
                # Already a standard FunctionSchema
                standard_functions.append(func)
            else:
                # Convert legacy dictionary format to FunctionSchema
                flows_schema = self.convert_to_function_schema(func)
                # Extract just the FunctionSchema part for the LLM
                standard_functions.append(
                    FunctionSchema(
                        name=flows_schema.name,
                        description=flows_schema.description,
                        properties=flows_schema.properties,
                        required=flows_schema.required,
                    )
                )

        # Return empty list to mean "no tools" if no valid functions were
        # processed
        if not standard_functions:
            return []

        # Create ToolsSchema with all functions
        return ToolsSchema(standard_tools=standard_functions)

    def format_summary_message(self, summary: str) -> dict:
        """Format a summary as a message appropriate for this LLM provider.

        Note: summary messages are always expected in OpenAI format, because
        summarization triggers an LLMMessagesUpdateFrame, which expects
        OpenAI-style messages.

        Args:
            summary: The generated summary text.

        Returns:
            A properly formatted message for this provider.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def generate_summary(
        self, llm: Any, summary_prompt: str, context: OpenAILLMContext | LLMContext
    ) -> Optional[str]:
        """Generate a summary by running a direct one-shot, out-of-band inference with the LLM.

        Args:
            llm: LLM service instance containing client/credentials.
            summary_prompt: Prompt text to guide summary generation.
            context: Context object containing conversation history for the summary.

        Returns:
            Generated summary text, or None if generation fails.
        """
        try:
            if isinstance(context, LLMContext):
                messages = context.get_messages()
            else:
                messages = context.messages

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

            summary_context = LLMContext(messages=prompt_messages)

            return await llm.run_inference(summary_context)

        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            return None

    def convert_to_function_schema(self, function_def: Dict[str, Any]) -> FlowsFunctionSchema:
        """Convert a provider-specific function definition to FlowsFunctionSchema.

        Args:
            function_def: Provider-specific function definition.

        Returns:
            FlowsFunctionSchema equivalent.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")


class UniversalLLMAdapter(LLMAdapter):
    """Format adapter when universal LLMContext is in use.

    Disallows provider-specific function definitions and requires functions to
    be in the standard FlowsFunctionSchema format.
    """

    def _get_function_name_from_dict(self, function_def: Dict[str, Any]) -> str:
        raise RuntimeError(
            "Provider-specific function definitions are not supported in flows using universal LLMContext. Use FlowsFunctionSchemas or direct functions instead."
        )

    def format_functions(
        self,
        functions: List[Union[Dict[str, Any], FunctionSchema, FlowsFunctionSchema]],
        original_configs: Optional[List] = None,
    ) -> ToolsSchema | NotGiven:
        """Format functions for flows using universal LLMContext.

        Args:
            functions: List of function definitions (dicts or schema objects).
            original_configs: Optional original node configs, used by some adapters.

        Returns:
            Functions formatted for use by universal LLMContext.
        """
        formatted_functions = super().format_functions(functions, original_configs=original_configs)
        # Ensure we only return ToolsSchema or NOT_GIVEN, as expected by
        # Pipecat's universal LLMContext.
        if not formatted_functions:
            return NOT_GIVEN
        return formatted_functions

    def format_summary_message(self, summary: str) -> dict:
        """Format summary as a system message for flows using universal LLMContext.

        Args:
            summary: The generated summary text.

        Returns:
            A system message containing the summary.
        """
        # The standard format in flows using universal LLMContext is the
        # LLMContextMessage format, which is based on OpenAI
        return {"role": "system", "content": f"Here's a summary of the conversation:\n{summary}"}

    def convert_to_function_schema(self, function_def):
        """Convert function definition to FlowsFunctionSchema.

        This should never be invoked in flows using universal LLMContext, as
        they should only use the standard FlowsFunctionSchema format.
        """
        raise RuntimeError(
            "Provider-specific function definitions are not supported in flows using universal LLMContext. Use FlowsFunctionSchemas or direct functions instead."
        )


class OpenAIAdapter(LLMAdapter):
    """Format adapter for OpenAI.

    Handles OpenAI's function calling format, which is used as the default format
    in the flow system.
    """

    def _get_function_name_from_dict(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from OpenAI function definition.

        Args:
            function_def: OpenAI-formatted function definition dictionary.

        Returns:
            Function name from the definition.
        """
        return function_def["function"]["name"]

    def format_summary_message(self, summary: str) -> dict:
        """Format summary as a system message for OpenAI.

        Args:
            summary: The generated summary text.

        Returns:
            OpenAI-formatted system message containing the summary.
        """
        return {"role": "system", "content": f"Here's a summary of the conversation:\n{summary}"}

    def convert_to_function_schema(self, function_def: Dict[str, Any]) -> FlowsFunctionSchema:
        """Convert OpenAI function definition to FlowsFunctionSchema.

        Args:
            function_def: OpenAI function definition.

        Returns:
            FlowsFunctionSchema equivalent with flow-specific fields.
        """
        func_data = function_def["function"]
        name = func_data["name"]
        description = func_data.get("description", "")
        parameters = func_data.get("parameters", {}) or {}
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        # Extract Flows-specific fields
        handler = func_data.get("handler")
        transition_to = func_data.get("transition_to")
        transition_callback = func_data.get("transition_callback")

        return FlowsFunctionSchema(
            name=name,
            description=description,
            properties=properties,
            required=required,
            handler=handler,
            transition_to=transition_to,
            transition_callback=transition_callback,
        )


class AnthropicAdapter(LLMAdapter):
    """Format adapter for Anthropic.

    Handles Anthropic's native function format, converting between OpenAI's format
    and Anthropic's as needed.
    """

    def _get_function_name_from_dict(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from Anthropic function definition.

        Args:
            function_def: Anthropic-formatted function definition dictionary.

        Returns:
            Function name from the definition.
        """
        return function_def["name"]

    def format_summary_message(self, summary: str) -> dict:
        """Format summary as a user message for Anthropic.

        Note: summary messages are always expected in OpenAI format, because
        summarization triggers an LLMMessagesUpdateFrame, which expects
        OpenAI-style messages.

        Args:
            summary: The generated summary text.

        Returns:
            Anthropic-formatted user message containing the summary.
        """
        return {"role": "user", "content": f"Here's a summary of the conversation:\n{summary}"}

    def convert_to_function_schema(self, function_def: Dict[str, Any]) -> FlowsFunctionSchema:
        """Convert Anthropic function definition to FlowsFunctionSchema.

        Args:
            function_def: Anthropic function definition.

        Returns:
            FlowsFunctionSchema equivalent with flow-specific fields.
        """
        name = function_def["name"]
        description = function_def.get("description", "")
        input_schema = function_def.get("input_schema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # Extract Flows-specific fields
        handler = function_def.get("handler")
        transition_to = function_def.get("transition_to")
        transition_callback = function_def.get("transition_callback")

        return FlowsFunctionSchema(
            name=name,
            description=description,
            properties=properties,
            required=required,
            handler=handler,
            transition_to=transition_to,
            transition_callback=transition_callback,
        )


class GeminiAdapter(LLMAdapter):
    """Format adapter for Google's Gemini.

    Handles Gemini's function declarations format, converting between OpenAI's format
    and Gemini's as needed.
    """

    def _get_function_name_from_dict(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from Gemini function definition.

        Args:
            function_def: Gemini-formatted function definition dictionary.

        Returns:
            Function name from the first declaration, or empty string if none found.
        """
        logger.debug(f"Getting function name from: {function_def}")
        if "function_declarations" in function_def:
            declarations = function_def["function_declarations"]
            if declarations and isinstance(declarations, list):
                return declarations[0]["name"]
        return ""

    def format_functions(
        self,
        functions: List[Union[Dict[str, Any], FunctionSchema, FlowsFunctionSchema]],
        original_configs: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """Format functions for Gemini's specific use.

        This special implementation processes both converted schemas and original configs
        to ensure Gemini's specific format is preserved when possible.

        Args:
            functions: List of function definitions (dicts or schema objects).
            original_configs: Optional original node configs, used to preserve native formats.

        Returns:
            List of functions formatted for Gemini.
        """
        # TODO: here we should always be formatting into FunctionSchemas, not provider-specific format
        gemini_functions = []

        # If original_configs is provided, extract functions from it
        if original_configs:
            for func_config in original_configs:
                if isinstance(func_config, FlowsFunctionSchema):
                    # Convert FlowsFunctionSchema to Gemini format
                    gemini_functions.append(
                        {
                            "name": func_config.name,
                            "description": func_config.description,
                            "parameters": {
                                "type": "object",
                                "properties": func_config.properties,
                                "required": func_config.required,
                            },
                        }
                    )
                elif isinstance(func_config, Callable):
                    # Convert direct function to Gemini format
                    direct_func = FlowsDirectFunctionWrapper(function=func_config)
                    gemini_functions.append(
                        {
                            "name": direct_func.name,
                            "description": direct_func.description,
                            "parameters": {
                                "type": "object",
                                "properties": direct_func.properties,
                                "required": direct_func.required,
                            },
                        }
                    )
                elif "function_declarations" in func_config:
                    # Already in Gemini format, use directly but remove handler/transition fields
                    for decl in func_config["function_declarations"]:
                        decl_copy = decl.copy()
                        if "handler" in decl_copy:
                            del decl_copy["handler"]
                        if "transition_to" in decl_copy:
                            del decl_copy["transition_to"]
                        if "transition_callback" in decl_copy:
                            del decl_copy["transition_callback"]
                        gemini_functions.append(decl_copy)
        else:
            # If no original configs, use the converted schemas
            for func in functions:
                if isinstance(func, (FunctionSchema, FlowsFunctionSchema)):
                    # Convert to Gemini format
                    gemini_functions.append(
                        {
                            "name": func.name,
                            "description": func.description,
                            "parameters": {
                                "type": "object",
                                "properties": func.properties,
                                "required": func.required,
                            },
                        }
                    )
        # Return empty list if no functions
        if not gemini_functions:
            return []

        # Format as Gemini expects - an array with a single object containing function_declarations
        return [{"function_declarations": gemini_functions}]

    def format_summary_message(self, summary: str) -> dict:
        """Format summary as a user message for Gemini.

        Note: summary messages are always expected in OpenAI format, because
        summarization triggers an LLMMessagesUpdateFrame, which expects
        OpenAI-style messages.

        Args:
            summary: The generated summary text.

        Returns:
            Gemini-formatted user message containing the summary.
        """
        return {"role": "user", "content": f"Here's a summary of the conversation:\n{summary}"}

    def convert_to_function_schema(self, function_def: Dict[str, Any]) -> FlowsFunctionSchema:
        """Convert Gemini function definition to FlowsFunctionSchema.

        Args:
            function_def: Gemini function definition.

        Returns:
            FlowsFunctionSchema equivalent with flow-specific fields.
        """
        if "function_declarations" in function_def:
            # Use first declaration if there are multiple
            decl = function_def["function_declarations"][0]
            # If we have function declarations, the handler might be in the declaration
            handler = decl.get("handler")
            transition_to = decl.get("transition_to")
            transition_callback = decl.get("transition_callback")
        else:
            decl = function_def
            # Otherwise, the handler might be at the top level
            handler = function_def.get("handler")
            transition_to = function_def.get("transition_to")
            transition_callback = function_def.get("transition_callback")

        name = decl["name"]
        description = decl.get("description", "")
        parameters = decl.get("parameters", {}) or {}
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        return FlowsFunctionSchema(
            name=name,
            description=description,
            properties=properties,
            required=required,
            handler=handler,
            transition_to=transition_to,
            transition_callback=transition_callback,
        )


class AWSBedrockAdapter(LLMAdapter):
    """Format adapter for AWS Bedrock.

    Handles both Anthropic Claude and Amazon Nova models on AWS Bedrock,
    converting between OpenAI's format and Bedrock's as needed.
    """

    def _get_function_name_from_dict(self, function_def: Dict[str, Any]) -> str:
        """Extract function name from Bedrock function definition.

        Args:
            function_def: Bedrock-formatted function definition dictionary.

        Returns:
            Function name from the definition.
        """
        # Bedrock uses the same format as Anthropic for tools
        return function_def["name"]

    def format_summary_message(self, summary: str) -> dict:
        """Format summary as a user message for Bedrock models.

        Note: summary messages are always expected in OpenAI format, because
        summarization triggers an LLMMessagesUpdateFrame, which expects
        OpenAI-style messages.

        Args:
            summary: The generated summary text.

        Returns:
            Bedrock-formatted user message containing the summary.
        """
        return {"role": "user", "content": f"Here's a summary of the conversation:\n{summary}"}

    def convert_to_function_schema(self, function_def: Dict[str, Any]) -> FlowsFunctionSchema:
        """Convert Bedrock function definition to FlowsFunctionSchema.

        Args:
            function_def: Bedrock function definition.

        Returns:
            FlowsFunctionSchema equivalent with flow-specific fields.
        """
        # Initialize with default values
        name = ""
        description = ""
        properties = {}
        required = []

        # Check for toolSpec format first
        if "toolSpec" in function_def:
            # Handle toolSpec format
            tool_spec = function_def["toolSpec"]
            name = tool_spec.get("name", "")
            description = tool_spec.get("description", "")
            input_schema = tool_spec.get("inputSchema", {})
            if "json" in input_schema:
                schema = input_schema["json"]
                properties = schema.get("properties", {})
                required = schema.get("required", [])
        # Handle standard formats
        else:
            name = function_def["name"]  # Now safe to access after checking toolSpec
            description = function_def.get("description", "")

            # Handle both possible schema formats
            if "input_schema" in function_def:
                input_schema = function_def.get("input_schema", {})
                if "json" in input_schema:
                    # Handle nested json schema format
                    schema = input_schema["json"]
                    properties = schema.get("properties", {})
                    required = schema.get("required", [])
                else:
                    # Handle direct schema format
                    properties = input_schema.get("properties", {})
                    required = input_schema.get("required", [])

        # Extract Flows-specific fields
        handler = function_def.get("handler")
        transition_to = function_def.get("transition_to")
        transition_callback = function_def.get("transition_callback")

        return FlowsFunctionSchema(
            name=name,
            description=description,
            properties=properties,
            required=required,
            handler=handler,
            transition_to=transition_to,
            transition_callback=transition_callback,
        )


def create_adapter(llm, context_aggregator) -> LLMAdapter:
    """Create appropriate adapter based on context type and LLM service type or inheritance.

    Checks both direct class types and inheritance hierarchies to determine
    the appropriate adapter for any LLM service.

    Args:
        llm: LLM service instance or LLMSwitcher.
        context_aggregator: Context aggregator pair.

    Returns:
        Provider-specific adapter instance.

    Raises:
        ValueError: If LLM type is not supported or required dependency not installed.
    """
    if isinstance(context_aggregator, LLMContextAggregatorPair):
        # Universal LLMContext is in use, so we need the universal adapter
        logger.debug("Creating universal adapter")
        return UniversalLLMAdapter()

    llm_type = type(llm).__name__
    llm_class = type(llm)

    if llm_type == "OpenAILLMService":
        logger.debug("Creating OpenAI adapter")
        return OpenAIAdapter()

    if llm_type == "AnthropicLLMService":
        logger.debug("Creating Anthropic adapter")
        return AnthropicAdapter()

    if llm_type == "GoogleLLMService":
        logger.debug("Creating Google adapter")
        return GeminiAdapter()

    if llm_type == "AWSBedrockLLMService":
        logger.debug("Creating Bedrock adapter")
        return AWSBedrockAdapter()

    # Try to find OpenAILLMService for inheritance check
    try:
        module = sys.modules.get("pipecat.services.openai")
        if module:
            openai_service = getattr(module, "OpenAILLMService", None)
            if openai_service and issubclass(llm_class, openai_service):
                logger.debug(f"Creating OpenAI adapter for {llm_type}")
                return OpenAIAdapter()
    except (TypeError, AttributeError) as e:
        # Log but continue to error handling if issubclass check fails
        logger.warning(f"Error checking inheritance for {llm_type}: {str(e)}")

    # Error handling
    error_msg = (
        f"Unsupported LLM type or missing dependency: {llm_type} (module: {llm_class.__module__})\n"
    )
    error_msg += "Make sure you have installed the required dependency:\n"
    error_msg += "- For OpenAI: pip install 'pipecat-ai[openai]'\n"
    error_msg += "- For Anthropic: pip install 'pipecat-ai[anthropic]'\n"
    error_msg += "- For Google: pip install 'pipecat-ai[google]'\n"
    error_msg += "- For Bedrock: pip install 'pipecat-ai[aws]'"

    raise ValueError(error_msg)
