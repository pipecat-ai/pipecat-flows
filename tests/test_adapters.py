#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for LLM provider adapters.

This module tests the adapter classes that normalize function formats between
Pipecat Flows and different LLM providers (OpenAI, Anthropic, Gemini, and AWS Bedrock).

Tests:
    - Native format handling for each provider
    - FlowsFunctionSchema handling for each provider
    - Function format conversions
    - Flow-specific field management

Mocks:
    - Provider-specific LLM adapters to avoid network calls
"""

import pytest
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from pipecat_flows.adapters import (
    AnthropicAdapter,
    AWSBedrockAdapter,
    GeminiAdapter,
    OpenAIAdapter,
)
from pipecat_flows.types import FlowsFunctionSchema


# Mock OpenAI's adapter to avoid actual network calls
class MockOpenAILLMAdapter:
    def to_provider_tools_format(self, tools_schema):
        # Simple mock that returns standard tools as OpenAI expects them
        return [
            {"type": "function", "function": func.to_default_dict()}
            for func in tools_schema.standard_tools
        ]


# Fixture for the adapter
@pytest.fixture
def openai_adapter():
    adapter = OpenAIAdapter()
    adapter._provider_adapter = MockOpenAILLMAdapter()
    return adapter


def test_openai_adapter_native_format(openai_adapter):
    """Test OpenAI adapter properly handles native OpenAI format."""
    # Create a function in OpenAI's native format
    openai_function = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature",
                    },
                },
                "required": ["location"],
            },
            "handler": lambda x: x,
            "transition_to": "next_step",
        },
    }

    # Get function name from dictionary
    function_name = openai_adapter.get_function_name(openai_function)
    assert function_name == "get_weather"

    # Convert to FlowsFunctionSchema
    schema = openai_adapter.convert_to_function_schema(openai_function)
    assert isinstance(schema, FlowsFunctionSchema)
    assert schema.name == "get_weather"
    assert schema.description == "Get the current weather in a location"
    assert "location" in schema.properties
    assert "unit" in schema.properties
    assert schema.required == ["location"]
    assert schema.handler is not None
    assert schema.transition_to == "next_step"
    assert schema.transition_callback is None

    # Format function for OpenAI (ToolsSchema expected)
    formatted = openai_adapter.format_functions([openai_function])
    assert isinstance(formatted, ToolsSchema)
    assert len(formatted.standard_tools) == 1
    assert formatted.standard_tools[0].name == "get_weather"


def test_openai_adapter_function_schema(openai_adapter):
    """Test OpenAI adapter properly handles FlowsFunctionSchema."""
    # Create a FlowsFunctionSchema
    flows_schema = FlowsFunctionSchema(
        name="get_weather",
        description="Get the current weather in a location",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature",
            },
        },
        required=["location"],
        handler=lambda x: x,
        transition_to="next_step",
    )

    # Get function name from schema
    function_name = openai_adapter.get_function_name(flows_schema)
    assert function_name == "get_weather"

    # Format schema for OpenAI (ToolsSchema expected)
    formatted = openai_adapter.format_functions([flows_schema])
    assert isinstance(formatted, ToolsSchema)
    assert len(formatted.standard_tools) == 1
    assert formatted.standard_tools[0].name == "get_weather"
    assert formatted.standard_tools[0].description == "Get the current weather in a location"
    assert "location" in formatted.standard_tools[0].properties
    assert "unit" in formatted.standard_tools[0].properties
    assert formatted.standard_tools[0].required == ["location"]


# Mock Anthropic's adapter to avoid actual network calls
class MockAnthropicLLMAdapter:
    def to_provider_tools_format(self, tools_schema):
        # Simple mock that returns standard tools as Anthropic expects them
        return [
            {
                "name": func.name,
                "description": func.description,
                "input_schema": {
                    "type": "object",
                    "properties": func.properties,
                    "required": func.required,
                },
            }
            for func in tools_schema.standard_tools
        ]


# Fixture for the adapter
@pytest.fixture
def anthropic_adapter():
    adapter = AnthropicAdapter()
    adapter._provider_adapter = MockAnthropicLLMAdapter()
    return adapter


def test_anthropic_adapter_native_format(anthropic_adapter):
    """Test Anthropic adapter properly handles native Anthropic format."""
    # Create a function in Anthropic's native format
    anthropic_function = {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature",
                },
            },
            "required": ["location"],
        },
        "handler": lambda x: x,
        "transition_callback": lambda x, y: None,
    }

    # Get function name from dictionary
    function_name = anthropic_adapter.get_function_name(anthropic_function)
    assert function_name == "get_weather"

    # Convert to FlowsFunctionSchema
    schema = anthropic_adapter.convert_to_function_schema(anthropic_function)
    assert isinstance(schema, FlowsFunctionSchema)
    assert schema.name == "get_weather"
    assert schema.description == "Get the current weather in a location"
    assert "location" in schema.properties
    assert "unit" in schema.properties
    assert schema.required == ["location"]
    assert schema.handler is not None
    assert schema.transition_to is None
    assert schema.transition_callback is not None

    # Format function for Anthropic (ToolsSchema expected)
    formatted = anthropic_adapter.format_functions([anthropic_function])
    assert isinstance(formatted, ToolsSchema)
    assert len(formatted.standard_tools) == 1
    assert formatted.standard_tools[0].name == "get_weather"
    assert formatted.standard_tools[0].description == "Get the current weather in a location"
    assert "location" in formatted.standard_tools[0].properties
    assert "unit" in formatted.standard_tools[0].properties
    assert formatted.standard_tools[0].required == ["location"]


def test_anthropic_adapter_function_schema(anthropic_adapter):
    """Test Anthropic adapter properly handles FlowsFunctionSchema."""
    # Create a FlowsFunctionSchema
    flows_schema = FlowsFunctionSchema(
        name="get_weather",
        description="Get the current weather in a location",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature",
            },
        },
        required=["location"],
        handler=lambda x: x,
        transition_callback=lambda x, y: None,
    )

    # Get function name from schema
    function_name = anthropic_adapter.get_function_name(flows_schema)
    assert function_name == "get_weather"

    # Format schema for Anthropic (ToolsSchema expected)
    formatted = anthropic_adapter.format_functions([flows_schema])
    assert isinstance(formatted, ToolsSchema)
    assert len(formatted.standard_tools) == 1
    assert formatted.standard_tools[0].name == "get_weather"
    assert formatted.standard_tools[0].description == "Get the current weather in a location"
    assert "location" in formatted.standard_tools[0].properties
    assert "unit" in formatted.standard_tools[0].properties
    assert formatted.standard_tools[0].required == ["location"]


# Mock Gemini's adapter to avoid actual network calls
class MockGeminiLLMAdapter:
    def to_provider_tools_format(self, tools_schema):
        # Simple mock that returns standard tools as Gemini expects them
        function_declarations = [
            {
                "name": func.name,
                "description": func.description,
                "parameters": {
                    "type": "object",
                    "properties": func.properties,
                    "required": func.required,
                },
            }
            for func in tools_schema.standard_tools
        ]
        return [{"function_declarations": function_declarations}]


# Fixture for the adapter
@pytest.fixture
def gemini_adapter():
    adapter = GeminiAdapter()
    adapter._provider_adapter = MockGeminiLLMAdapter()
    return adapter


def test_gemini_adapter_native_format(gemini_adapter):
    """Test Gemini adapter properly handles native Gemini format."""
    # Create a function in Gemini's native format
    gemini_function = {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature",
                        },
                    },
                    "required": ["location"],
                },
                "handler": lambda x: x,
                "transition_to": "next_step",
            }
        ]
    }

    # Get function name from dictionary
    function_name = gemini_adapter.get_function_name(gemini_function)
    assert function_name == "get_weather"

    # Convert to FlowsFunctionSchema
    schema = gemini_adapter.convert_to_function_schema(gemini_function)
    assert isinstance(schema, FlowsFunctionSchema)
    assert schema.name == "get_weather"
    assert schema.description == "Get the current weather in a location"
    assert "location" in schema.properties
    assert "unit" in schema.properties
    assert schema.required == ["location"]
    assert schema.handler is not None
    assert schema.transition_to == "next_step"
    assert schema.transition_callback is None

    # Format function for Gemini - using the specific format_functions implementation
    formatted = gemini_adapter.format_functions([gemini_function], [gemini_function])
    assert len(formatted) == 1
    assert "function_declarations" in formatted[0]
    assert len(formatted[0]["function_declarations"]) == 1
    assert formatted[0]["function_declarations"][0]["name"] == "get_weather"

    # Verify flow-specific fields not in formatted output
    assert "handler" not in formatted[0]["function_declarations"][0]
    assert "transition_to" not in formatted[0]["function_declarations"][0]


def test_gemini_adapter_function_schema(gemini_adapter):
    """Test Gemini adapter properly handles FlowsFunctionSchema."""
    # Create a FlowsFunctionSchema
    flows_schema = FlowsFunctionSchema(
        name="get_weather",
        description="Get the current weather in a location",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature",
            },
        },
        required=["location"],
        handler=lambda x: x,
        transition_to="next_step",
    )

    # Get function name from schema
    function_name = gemini_adapter.get_function_name(flows_schema)
    assert function_name == "get_weather"

    # Format schema for Gemini
    formatted = gemini_adapter.format_functions([flows_schema])
    assert len(formatted) == 1
    assert "function_declarations" in formatted[0]
    assert len(formatted[0]["function_declarations"]) == 1
    assert formatted[0]["function_declarations"][0]["name"] == "get_weather"
    assert (
        formatted[0]["function_declarations"][0]["description"]
        == "Get the current weather in a location"
    )

    # Verify flow-specific fields not in formatted output
    assert "handler" not in formatted[0]["function_declarations"][0]
    assert "transition_to" not in formatted[0]["function_declarations"][0]


# Mock AWS Bedrock adapter to avoid actual network calls
class MockAWSBedrockLLMAdapter:
    def to_provider_tools_format(self, tools_schema):
        # Simple mock that returns standard tools as AWS Bedrock expects them (using Claude format)
        return [
            {
                "name": func.name,
                "description": func.description,
                "input_schema": {
                    "json": {
                        "type": "object",
                        "properties": func.properties,
                        "required": func.required,
                    }
                },
            }
            for func in tools_schema.standard_tools
        ]


# Fixture for the AWS Bedrock adapter
@pytest.fixture
def bedrock_adapter():
    adapter = AWSBedrockAdapter()
    adapter._provider_adapter = MockAWSBedrockLLMAdapter()
    return adapter


def test_bedrock_adapter_native_format(bedrock_adapter):
    """Test AWS Bedrock adapter properly handles native Bedrock format."""
    # Create a function in Bedrock's native format (Claude-style)
    bedrock_function = {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "input_schema": {
            "json": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature",
                    },
                },
                "required": ["location"],
            }
        },
        "handler": lambda x: x,
        "transition_to": "next_step",
    }

    # Get function name from dictionary
    function_name = bedrock_adapter.get_function_name(bedrock_function)
    assert function_name == "get_weather"

    # Convert to FlowsFunctionSchema
    schema = bedrock_adapter.convert_to_function_schema(bedrock_function)
    assert isinstance(schema, FlowsFunctionSchema)
    assert schema.name == "get_weather"
    assert schema.description == "Get the current weather in a location"
    assert "location" in schema.properties
    assert "unit" in schema.properties
    assert schema.required == ["location"]
    assert schema.handler is not None
    assert schema.transition_to == "next_step"
    assert schema.transition_callback is None

    # Format function for Bedrock (ToolsSchema expected)
    formatted = bedrock_adapter.format_functions([bedrock_function])
    assert isinstance(formatted, ToolsSchema)
    assert len(formatted.standard_tools) == 1
    assert formatted.standard_tools[0].name == "get_weather"
    assert formatted.standard_tools[0].description == "Get the current weather in a location"
    assert "location" in formatted.standard_tools[0].properties


def test_bedrock_adapter_function_schema(bedrock_adapter):
    """Test AWS Bedrock adapter properly handles FlowsFunctionSchema."""
    # Create a FlowsFunctionSchema
    flows_schema = FlowsFunctionSchema(
        name="get_weather",
        description="Get the current weather in a location",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature",
            },
        },
        required=["location"],
        handler=lambda x: x,
        transition_callback=lambda x, y: None,
    )

    # Get function name from schema
    function_name = bedrock_adapter.get_function_name(flows_schema)
    assert function_name == "get_weather"

    # Format schema for Bedrock (ToolsSchema expected)
    formatted = bedrock_adapter.format_functions([flows_schema])
    assert isinstance(formatted, ToolsSchema)
    assert len(formatted.standard_tools) == 1
    assert formatted.standard_tools[0].name == "get_weather"
    assert formatted.standard_tools[0].description == "Get the current weather in a location"
    assert "location" in formatted.standard_tools[0].properties
    assert "unit" in formatted.standard_tools[0].properties
    assert formatted.standard_tools[0].required == ["location"]


def test_bedrock_adapter_toolspec_format(bedrock_adapter):
    """Test AWS Bedrock adapter properly handles the toolSpec format."""
    # Create a function using the Bedrock toolSpec format (alternative format)
    bedrock_toolspec_function = {
        "toolSpec": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature",
                        },
                    },
                    "required": ["location"],
                }
            },
        },
        "handler": lambda x: x,
        "transition_to": "next_step",
    }

    # Convert to FlowsFunctionSchema
    schema = bedrock_adapter.convert_to_function_schema(bedrock_toolspec_function)
    assert isinstance(schema, FlowsFunctionSchema)
    assert schema.name == "get_weather"
    assert schema.description == "Get the current weather in a location"
    assert "location" in schema.properties
    assert "unit" in schema.properties
    assert schema.required == ["location"]
    assert schema.handler is not None
    assert schema.transition_to == "next_step"
    assert schema.transition_callback is None


def test_openai_adapter_empty_functions(openai_adapter):
    """Test OpenAI adapter properly handles empty function arrays."""
    # Format empty list for OpenAI
    formatted = openai_adapter.format_functions([])
    # OpenAI supports empty function arrays
    assert formatted == []


def test_anthropic_adapter_empty_functions(anthropic_adapter):
    """Test Anthropic adapter properly handles empty function arrays."""
    # Format empty list for Anthropic
    formatted = anthropic_adapter.format_functions([])
    # OpenAI supports empty function arrays
    assert formatted == []


def test_gemini_adapter_empty_functions(gemini_adapter):
    """Test Gemini adapter properly handles empty function arrays."""
    # Format empty list for Gemini
    formatted = gemini_adapter.format_functions([])
    # OpenAI supports empty function arrays
    assert formatted == []


def test_bedrock_adapter_empty_functions(bedrock_adapter):
    """Test AWS Bedrock adapter properly handles empty function arrays."""
    # Format empty list for Bedrock
    formatted = bedrock_adapter.format_functions([])
    # OpenAI supports empty function arrays
    assert formatted == []
