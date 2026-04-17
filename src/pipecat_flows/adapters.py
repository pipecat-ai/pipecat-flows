#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM adapter for normalizing function and message formats.

This module provides the LLMAdapter class that normalizes interactions between
the flow manager and Pipecat's universal LLMContext. It handles:

- Function name extraction from FlowsFunctionSchema
- Function formatting into ToolsSchema for LLMSetToolsFrame
- Summary message formatting
- Summary generation via out-of-band LLM inference
"""

from typing import Any

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    NOT_GIVEN,
    LLMContext,
    LLMContextMessage,
    NotGiven,
)

from pipecat_flows.types import FlowsFunctionSchema


class LLMAdapter:
    """Adapter for normalizing function and message formats.

    Normalizes interactions between the flow manager and Pipecat's universal
    LLMContext. Functions must be provided as FlowsFunctionSchema or
    FunctionSchema objects.
    """

    def format_functions(
        self,
        functions: list[FunctionSchema | FlowsFunctionSchema],
    ) -> ToolsSchema | NotGiven:
        """Format functions into a ToolsSchema for use in LLMSetToolsFrame.

        Args:
            functions: List of function definitions (schema objects).

        Returns:
            ToolsSchema containing the functions, or NOT_GIVEN if no functions.
        """
        if not functions:
            return NOT_GIVEN

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

        if not standard_functions:
            return NOT_GIVEN

        return ToolsSchema(standard_tools=standard_functions)

    def format_summary_message(self, summary: str) -> dict:
        """Format a summary as a developer message.

        Summary messages use the LLMContextMessage format (OpenAI-style),
        as summarization triggers an LLMMessagesUpdateFrame.

        Args:
            summary: The generated summary text.

        Returns:
            A developer message containing the summary.
        """
        return {"role": "developer", "content": f"Here's a summary of the conversation:\n{summary}"}

    async def generate_summary(
        self, llm: Any, summary_prompt: str, context: LLMContext
    ) -> str | None:
        """Generate a summary by running a direct one-shot, out-of-band inference with the LLM.

        Args:
            llm: LLM service instance containing client/credentials.
            summary_prompt: Prompt text to guide summary generation.
            context: Context object containing conversation history for the summary.

        Returns:
            Generated summary text, or None if generation fails.
        """
        try:
            messages = context.get_messages()

            prompt_messages: list[LLMContextMessage] = [
                {
                    "role": "developer",
                    "content": f"Conversation history: {messages}",
                },
            ]

            summary_context = LLMContext(messages=prompt_messages)

            return await llm.run_inference(summary_context, system_instruction=summary_prompt)

        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            return None
