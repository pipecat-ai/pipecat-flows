#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the LLM adapter.

This module tests the LLMAdapter class that normalizes function formats
between Pipecat Flows and Pipecat's universal LLMContext.

Tests:
    - FlowsFunctionSchema handling
    - Function formatting into ToolsSchema
    - Empty function handling
    - Summary message formatting
"""

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import NOT_GIVEN

from pipecat_flows.adapters import LLMAdapter
from pipecat_flows.types import FlowsFunctionSchema


@pytest.fixture
def adapter():
    return LLMAdapter()


def test_format_functions_with_schema(adapter):
    """Test formatting FlowsFunctionSchema into ToolsSchema."""
    schema = FlowsFunctionSchema(
        name="get_weather",
        description="Get the current weather in a location",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state",
            },
        },
        required=["location"],
        handler=lambda x: x,
    )

    formatted = adapter.format_functions([schema])
    assert isinstance(formatted, ToolsSchema)
    assert len(formatted.standard_tools) == 1
    assert formatted.standard_tools[0].name == "get_weather"
    assert formatted.standard_tools[0].description == "Get the current weather in a location"
    assert "location" in formatted.standard_tools[0].properties
    assert formatted.standard_tools[0].required == ["location"]


def test_format_functions_with_function_schema(adapter):
    """Test formatting standard FunctionSchema into ToolsSchema."""
    schema = FunctionSchema(
        name="get_weather",
        description="Get weather",
        properties={"location": {"type": "string"}},
        required=["location"],
    )

    formatted = adapter.format_functions([schema])
    assert isinstance(formatted, ToolsSchema)
    assert len(formatted.standard_tools) == 1
    assert formatted.standard_tools[0].name == "get_weather"


def test_format_functions_empty(adapter):
    """Test that empty function list returns NOT_GIVEN."""
    formatted = adapter.format_functions([])
    assert formatted is NOT_GIVEN


def test_format_functions_multiple(adapter):
    """Test formatting multiple functions into ToolsSchema."""
    schemas = [
        FlowsFunctionSchema(
            name="get_weather",
            description="Get weather",
            properties={},
            required=[],
        ),
        FlowsFunctionSchema(
            name="get_time",
            description="Get time",
            properties={},
            required=[],
        ),
    ]

    formatted = adapter.format_functions(schemas)
    assert isinstance(formatted, ToolsSchema)
    assert len(formatted.standard_tools) == 2
    assert formatted.standard_tools[0].name == "get_weather"
    assert formatted.standard_tools[1].name == "get_time"


def test_format_summary_message(adapter):
    """Test summary message formatting."""
    message = adapter.format_summary_message("Test summary")
    assert message == {
        "role": "developer",
        "content": "Here's a summary of the conversation:\nTest summary",
    }
