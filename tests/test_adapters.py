import unittest
from unittest.mock import MagicMock

from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.google import GoogleLLMService
from pipecat.services.openai import OpenAILLMService

from pipecat_flows.adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    LLMAdapter,
    OpenAIAdapter,
    create_adapter,
)


class TestLLMAdapter(unittest.TestCase):
    """Test the abstract base LLMAdapter class"""

    def test_abstract_methods(self):
        """Verify that LLMAdapter cannot be instantiated without implementing all methods"""

        class IncompleteAdapter(LLMAdapter):
            # Missing implementation of abstract methods
            pass

        with self.assertRaises(TypeError):
            IncompleteAdapter()

        class PartialAdapter(LLMAdapter):
            def get_function_name(self, function_def):
                return "test"

            # Still missing other required methods

        with self.assertRaises(TypeError):
            PartialAdapter()


class TestLLMAdapters(unittest.TestCase):
    def setUp(self):
        """Set up test cases with sample function definitions for each provider"""
        # OpenAI format
        self.openai_function = {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function",
                "parameters": {"type": "object", "properties": {"param1": {"type": "string"}}},
            },
        }

        self.openai_function_call = {"name": "test_function", "arguments": {"param1": "value1"}}

        # Anthropic format
        self.anthropic_function = {
            "name": "test_function",
            "description": "Test function",
            "input_schema": {"type": "object", "properties": {"param1": {"type": "string"}}},
        }

        self.anthropic_function_call = {"name": "test_function", "arguments": {"param1": "value1"}}

        # Gemini format
        self.gemini_function = {
            "function_declarations": [
                {
                    "name": "test_function",
                    "description": "Test function",
                    "parameters": {"type": "object", "properties": {"param1": {"type": "string"}}},
                }
            ]
        }

        self.gemini_function_call = {"name": "test_function", "args": {"param1": "value1"}}

        # Message formats
        self.openai_message = {"role": "system", "content": "Test message"}

        self.null_message = {"role": "system", "content": None}

        self.anthropic_message = {
            "role": "user",
            "content": [{"type": "text", "text": "Test message"}],
        }

        self.gemini_message = {"role": "user", "content": "Test message"}

    def test_openai_adapter(self):
        """Test OpenAI format handling"""
        adapter = OpenAIAdapter()

        # Test function name extraction
        self.assertEqual(adapter.get_function_name(self.openai_function), "test_function")

        # Test function arguments extraction
        args = adapter.get_function_args(self.openai_function_call)
        self.assertEqual(args, {"param1": "value1"})

        # Test message content extraction
        self.assertEqual(adapter.get_message_content(self.openai_message), "Test message")

        # Test null message content
        # The implementation returns None for null content
        self.assertIsNone(adapter.get_message_content(self.null_message))

        # Test function formatting
        formatted = adapter.format_functions([self.openai_function])
        self.assertEqual(formatted, [self.openai_function])

    def test_anthropic_adapter(self):
        """Test Anthropic format handling"""
        adapter = AnthropicAdapter()

        # Test function name extraction
        self.assertEqual(adapter.get_function_name(self.anthropic_function), "test_function")

        # Test function arguments extraction
        self.assertEqual(
            adapter.get_function_args(self.anthropic_function_call), {"param1": "value1"}
        )

        # Test message content extraction
        self.assertEqual(adapter.get_message_content(self.anthropic_message), "Test message")

        # Test function formatting
        formatted = adapter.format_functions([self.openai_function])
        self.assertTrue("input_schema" in formatted[0])
        self.assertEqual(formatted[0]["name"], "test_function")

    def test_gemini_adapter(self):
        """Test Gemini format handling"""
        adapter = GeminiAdapter()

        # Test function name extraction from function declarations
        self.assertEqual(
            adapter.get_function_name(self.gemini_function),  # Pass the full function object
            "test_function",
        )

        # Test function arguments extraction
        self.assertEqual(adapter.get_function_args(self.gemini_function_call), {"param1": "value1"})

        # Test message content extraction
        self.assertEqual(adapter.get_message_content(self.gemini_message), "Test message")

        # Test function formatting
        formatted = adapter.format_functions([self.openai_function])
        self.assertTrue("function_declarations" in formatted[0])

    def test_adapter_factory(self):
        """Test adapter creation based on LLM service type"""
        # Test with valid LLM services
        openai_llm = MagicMock(spec=OpenAILLMService)
        self.assertIsInstance(create_adapter(openai_llm), OpenAIAdapter)

        anthropic_llm = MagicMock(spec=AnthropicLLMService)
        self.assertIsInstance(create_adapter(anthropic_llm), AnthropicAdapter)

        gemini_llm = MagicMock(spec=GoogleLLMService)
        self.assertIsInstance(create_adapter(gemini_llm), GeminiAdapter)

    def test_adapter_factory_error_cases(self):
        """Test error cases in adapter creation"""
        # Test with None
        with self.assertRaises(ValueError) as context:
            create_adapter(None)
        self.assertIn("Unsupported LLM type", str(context.exception))

        # Test with invalid service type
        invalid_llm = MagicMock()
        with self.assertRaises(ValueError) as context:
            create_adapter(invalid_llm)
        self.assertIn("Unsupported LLM type", str(context.exception))

    def test_null_and_empty_values(self):
        """Test handling of null and empty values"""
        adapters = [OpenAIAdapter(), AnthropicAdapter(), GeminiAdapter()]

        for adapter in adapters:
            # Test empty function call
            empty_call = {"name": "test"}
            self.assertEqual(adapter.get_function_args(empty_call), {})

            # Test empty message
            empty_message = {"role": "user", "content": ""}
            self.assertEqual(adapter.get_message_content(empty_message), "")

    def test_special_characters_handling(self):
        """Test handling of special characters in messages and function calls"""
        special_chars = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/~`"

        # Test in message content
        message_with_special = {"role": "user", "content": f"Test with {special_chars}"}

        adapters = [OpenAIAdapter(), AnthropicAdapter(), GeminiAdapter()]
        for adapter in adapters:
            content = adapter.get_message_content(message_with_special)
            self.assertEqual(content, f"Test with {special_chars}")

        # Test in function arguments
        # Each adapter might handle arguments differently, so test them separately

        # OpenAI
        openai_adapter = OpenAIAdapter()
        openai_call = {"name": "test", "arguments": {"param1": special_chars}}
        args = openai_adapter.get_function_args(openai_call)
        self.assertEqual(args["param1"], special_chars)

        # Anthropic
        anthropic_adapter = AnthropicAdapter()
        anthropic_call = {"name": "test", "arguments": {"param1": special_chars}}
        args = anthropic_adapter.get_function_args(anthropic_call)
        self.assertEqual(args["param1"], special_chars)

        # Gemini
        gemini_adapter = GeminiAdapter()
        gemini_call = {
            "name": "test",
            "args": {"param1": special_chars},  # Note: Gemini uses 'args' instead of 'arguments'
        }
        args = gemini_adapter.get_function_args(gemini_call)
        self.assertEqual(args["param1"], special_chars)

    def test_function_schema_validation(self):
        """Test validation of function schemas during conversion"""
        adapters = [OpenAIAdapter(), AnthropicAdapter(), GeminiAdapter()]

        # Test with minimal valid schema
        minimal_function = {
            "type": "function",
            "function": {"name": "test", "parameters": {"type": "object", "properties": {}}},
        }

        for adapter in adapters:
            formatted = adapter.format_functions([minimal_function])
            self.assertTrue(len(formatted) > 0)

    def test_abstract_methods_implementation(self):
        """Test that abstract methods raise NotImplementedError."""

        class TestAdapter(LLMAdapter):
            pass  # No implementations

        with self.assertRaises(TypeError):
            TestAdapter()

        # Test partial implementation
        class PartialAdapter(LLMAdapter):
            def get_function_name(self, function_def):
                return ""

            # Missing other methods

        with self.assertRaises(TypeError):
            PartialAdapter()

    def test_anthropic_format_functions_passthrough(self):
        """Test Anthropic adapter passing through already formatted functions."""
        adapter = AnthropicAdapter()

        # Test with already formatted Anthropic function
        anthropic_formatted = {"name": "test", "description": "test", "input_schema": {}}

        result = adapter.format_functions([anthropic_formatted])
        self.assertEqual(result[0], anthropic_formatted)

    def test_gemini_adapter_empty_declarations(self):
        """Test Gemini adapter with empty or invalid declarations."""
        adapter = GeminiAdapter()

        # Test empty function declarations
        empty_decl = {"function_declarations": []}
        self.assertEqual(adapter.get_function_name(empty_decl), "")

        # Test invalid function declarations
        invalid_decl = {"function_declarations": None}
        self.assertEqual(adapter.get_function_name(invalid_decl), "")

        # Test missing function declarations
        missing_decl = {}
        self.assertEqual(adapter.get_function_name(missing_decl), "")

    def test_gemini_format_functions_full_schema(self):
        """Test Gemini adapter formatting with full schema."""
        adapter = GeminiAdapter()

        # Test with complete function declarations
        functions = [
            {
                "function_declarations": [
                    {
                        "name": "test1",
                        "description": "Test function 1",
                        "parameters": {
                            "type": "object",
                            "properties": {"param1": {"type": "string"}},
                        },
                    },
                    {
                        "name": "test2",
                        "description": "Test function 2",
                        "parameters": {
                            "type": "object",
                            "properties": {"param2": {"type": "number"}},
                        },
                    },
                ]
            }
        ]

        formatted = adapter.format_functions(functions)

        # Verify all declarations were formatted
        declarations = formatted[0]["function_declarations"]
        self.assertEqual(len(declarations), 2)

        # Verify first declaration
        self.assertEqual(declarations[0]["name"], "test1")
        self.assertEqual(declarations[0]["description"], "Test function 1")
        self.assertIn("parameters", declarations[0])

        # Verify second declaration
        self.assertEqual(declarations[1]["name"], "test2")
        self.assertEqual(declarations[1]["description"], "Test function 2")
        self.assertIn("parameters", declarations[1])

        # Test with minimal declarations (missing optional fields)
        minimal_functions = [
            {
                "function_declarations": [
                    {
                        "name": "test",
                        # Missing description and parameters
                    }
                ]
            }
        ]

        minimal_formatted = adapter.format_functions(minimal_functions)
        minimal_decl = minimal_formatted[0]["function_declarations"][0]

        # Verify defaults for missing fields
        self.assertEqual(minimal_decl["description"], "")
        self.assertEqual(minimal_decl["parameters"], {"type": "object", "properties": {}})
