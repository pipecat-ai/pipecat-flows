#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Test suite for context management strategies.

This module contains tests for the context management features of Pipecat Flows,
focusing on:
- Context strategy configuration
- Strategy behavior (APPEND, RESET, RESET_WITH_SUMMARY)
- Provider-specific message formatting
- Summary generation and integration
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from pipecat.frames.frames import LLMMessagesAppendFrame, LLMMessagesUpdateFrame
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_flows.exceptions import FlowError
from pipecat_flows.manager import FlowManager
from pipecat_flows.types import ContextStrategy, ContextStrategyConfig, NodeConfig


class TestContextStrategies(unittest.IsolatedAsyncioTestCase):
    """Test suite for context management strategies.

    Tests functionality including:
    - Strategy configuration and validation
    - Strategy behavior and message handling
    - Provider-specific adaptations
    - Summary generation and integration
    """

    async def asyncSetUp(self):
        """Set up test fixtures before each test."""
        self.mock_task = AsyncMock()
        self.mock_task.event_handler = Mock()
        self.mock_task.set_reached_downstream_filter = Mock()

        # Set up mock LLM with client
        self.mock_llm = OpenAILLMService(api_key="")
        self.mock_llm.run_inference = AsyncMock()

        self.mock_tts = AsyncMock()

        # Create mock context aggregator with messages
        self.mock_context = MagicMock()
        self.mock_context.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        self.mock_context_aggregator = MagicMock()
        self.mock_context_aggregator.user = MagicMock()
        self.mock_context_aggregator.user.return_value = MagicMock()
        self.mock_context_aggregator.user.return_value._context = self.mock_context

        # Sample node configuration
        self.sample_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test task."}],
            "functions": [],
        }

    async def test_context_strategy_config_validation(self):
        """Test ContextStrategyConfig validation."""
        # Valid configurations
        ContextStrategyConfig(strategy=ContextStrategy.APPEND)
        ContextStrategyConfig(strategy=ContextStrategy.RESET)
        ContextStrategyConfig(
            strategy=ContextStrategy.RESET_WITH_SUMMARY, summary_prompt="Summarize the conversation"
        )

        # Invalid configuration - missing prompt
        with self.assertRaises(ValueError):
            ContextStrategyConfig(strategy=ContextStrategy.RESET_WITH_SUMMARY)

    def _get_all_queued_frames(self):
        """Helper to collect all frames from all queue_frames calls."""
        all_frames = []
        for call in self.mock_task.queue_frames.call_args_list:
            frames = call[0][0]
            all_frames.extend(frames)
        return all_frames

    async def test_default_strategy(self):
        """Test default context strategy (APPEND)."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # First node should use UpdateFrame regardless of strategy
        await flow_manager._set_node("first", self.sample_node)
        first_frames = self._get_all_queued_frames()
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in first_frames))

        # Reset mock
        self.mock_task.queue_frames.reset_mock()

        # Subsequent node should use AppendFrame with default strategy
        await flow_manager._set_node("second", self.sample_node)
        second_frames = self._get_all_queued_frames()
        self.assertTrue(any(isinstance(f, LLMMessagesAppendFrame) for f in second_frames))

    async def test_reset_strategy(self):
        """Test RESET strategy behavior."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
        )
        await flow_manager.initialize()

        # Set initial node
        await flow_manager._set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        # Second node should use UpdateFrame with RESET strategy
        await flow_manager._set_node("second", self.sample_node)
        second_frames = self._get_all_queued_frames()
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames))

    async def test_reset_with_summary_success(self):
        """Test successful RESET_WITH_SUMMARY strategy."""
        # Mock successful summary generation
        mock_summary = "Conversation summary"
        self.mock_llm.run_inference.return_value = mock_summary

        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt="Summarize the conversation",
            ),
        )
        await flow_manager.initialize()

        # Set nodes and verify summary inclusion
        await flow_manager._set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        await flow_manager._set_node("second", self.sample_node)

        # Verify summary was included in context update
        second_frames = self._get_all_queued_frames()
        update_frame = next(f for f in second_frames if isinstance(f, LLMMessagesUpdateFrame))
        self.assertTrue(any(mock_summary in str(m) for m in update_frame.messages))

    async def test_reset_with_summary_timeout(self):
        """Test RESET_WITH_SUMMARY fallback to APPEND on timeout."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt="Summarize the conversation",
            ),
        )
        await flow_manager.initialize()

        # Mock timeout
        self.mock_llm.run_inference.side_effect = AsyncMock(side_effect=TimeoutError)

        # Set nodes and verify fallback to APPEND
        await flow_manager._set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        await flow_manager._set_node("second", self.sample_node)

        # Verify AppendFrame was used (fallback to APPEND behavior on timeout)
        second_frames = self._get_all_queued_frames()
        self.assertTrue(any(isinstance(f, LLMMessagesAppendFrame) for f in second_frames))

    async def test_provider_specific_summary_formatting(self):
        """Test summary formatting for different LLM providers."""
        summary = "Test summary"

        # Test OpenAI format
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=OpenAILLMService(api_key=""),
            context_aggregator=self.mock_context_aggregator,
        )
        openai_message = flow_manager._adapter.format_summary_message(summary)
        self.assertEqual(openai_message["role"], "system")

        # Test Anthropic format
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=AnthropicLLMService(api_key=""),
            context_aggregator=self.mock_context_aggregator,
        )
        anthropic_message = flow_manager._adapter.format_summary_message(summary)
        self.assertEqual(anthropic_message["role"], "user")

        # Test Gemini format
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=GoogleLLMService(api_key=" "),  # dummy key (GoogleLLMService rejects empty string)
            context_aggregator=self.mock_context_aggregator,
        )
        gemini_message = flow_manager._adapter.format_summary_message(summary)
        self.assertEqual(gemini_message["role"], "user")

    async def test_node_level_strategy_override(self):
        """Test that node-level strategy overrides global strategy."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        # Create node with RESET strategy
        node_with_strategy = {
            **self.sample_node,
            "context_strategy": ContextStrategyConfig(strategy=ContextStrategy.RESET),
        }

        # Set nodes and verify strategy override
        await flow_manager._set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        await flow_manager._set_node("second", node_with_strategy)

        # Verify UpdateFrame was used (RESET behavior) despite global APPEND
        second_frames = self._get_all_queued_frames()
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames))

    async def test_summary_generation_content(self):
        """Test that summary generation uses correct prompt and context."""
        mock_summary = "Generated summary"
        self.mock_llm.run_inference.return_value = mock_summary

        summary_prompt = "Create a detailed summary"
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY, summary_prompt=summary_prompt
            ),
        )
        await flow_manager.initialize()

        # Set nodes to trigger summary generation
        await flow_manager._set_node("first", self.sample_node)
        await flow_manager._set_node("second", self.sample_node)

        # Verify summary generation call
        run_inference_call = self.mock_llm.run_inference.call_args
        run_inference_args = run_inference_call[0]

        # Verify prompt and context were included
        context = run_inference_args[0]
        self.assertTrue(any(summary_prompt in str(m) for m in context.get_messages()))
        self.assertTrue(
            any(
                str(self.mock_context.messages[0]["content"]) in str(m)
                for m in context.get_messages()
            )
        )

    async def test_context_structure_after_summary(self):
        """Test the structure of context after summary generation."""
        mock_summary = "Generated summary"
        self.mock_llm.run_inference.return_value = mock_summary

        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY, summary_prompt="Summarize"
            ),
        )
        await flow_manager.initialize()

        # Set nodes to trigger summary generation
        await flow_manager._set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        # Node with new task messages
        new_node = {
            "task_messages": [{"role": "system", "content": "New task."}],
            "functions": [],
        }
        await flow_manager._set_node("second", new_node)

        # Verify context structure
        all_frames = self._get_all_queued_frames()
        messages_frame = next(
            (f for f in all_frames if isinstance(f, LLMMessagesUpdateFrame)), None
        )
        self.assertIsNotNone(messages_frame, "LLMMessagesUpdateFrame should be queued")

        # Verify order: summary message, then new task messages
        self.assertTrue(mock_summary in str(messages_frame.messages[0]))
        self.assertEqual(
            messages_frame.messages[1]["content"], new_node["task_messages"][0]["content"]
        )


class TestDeactivatedFunctions(unittest.IsolatedAsyncioTestCase):
    """Test suite for deactivated function carry-over during APPEND transitions.

    Tests functionality including:
    - Carrying over deactivated functions with dummy handlers
    - Injecting warning task messages
    - Skipping deactivation for RESET strategies
    - Global functions are not deactivated
    """

    async def asyncSetUp(self):
        """Set up test fixtures before each test."""
        self.mock_task = AsyncMock()
        self.mock_task.event_handler = Mock()
        self.mock_task.set_reached_downstream_filter = Mock()

        self.mock_llm = OpenAILLMService(api_key="")
        self.mock_llm.register_function = MagicMock()

        self.mock_context_aggregator = MagicMock()
        self.mock_context_aggregator.user = MagicMock()
        self.mock_context_aggregator.user.return_value = MagicMock()
        self.mock_context_aggregator.user.return_value._context = MagicMock()

    def _get_all_queued_frames(self):
        """Helper to collect all frames from all queue_frames calls."""
        all_frames = []
        for call in self.mock_task.queue_frames.call_args_list:
            frames = call[0][0]
            all_frames.extend(frames)
        return all_frames

    async def test_deactivated_functions_carried_over_on_append(self):
        """Test that functions from previous node are carried over as deactivated on APPEND."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        # First node with function_a and function_b
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Function A",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "function_b",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Function B",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)

        # Verify both functions are in current_functions
        self.assertIn("function_a", flow_manager._current_functions)
        self.assertIn("function_b", flow_manager._current_functions)

        self.mock_task.queue_frames.reset_mock()
        self.mock_llm.register_function.reset_mock()

        # Second node with only function_c (function_a and function_b should be deactivated)
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_c",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Function C",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("second", second_node)

        # Verify that function_a and function_b were registered as deactivated
        registered_function_names = [
            call[0][0] for call in self.mock_llm.register_function.call_args_list
        ]
        self.assertIn("function_a", registered_function_names)
        self.assertIn("function_b", registered_function_names)
        self.assertIn("function_c", registered_function_names)

    async def test_deactivated_warning_message_injected(self):
        """Test that warning message is injected when functions are deactivated."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        # First node with a function
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "old_function",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Old function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)
        self.mock_task.queue_frames.reset_mock()

        # Second node without the old function
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [],
        }

        await flow_manager._set_node("second", second_node)

        # Verify warning message was injected
        all_frames = self._get_all_queued_frames()
        messages_frame = next(
            (f for f in all_frames if isinstance(f, LLMMessagesAppendFrame)), None
        )
        self.assertIsNotNone(messages_frame, "LLMMessagesAppendFrame should be queued")

        # Check that warning message is present
        warning_found = any(
            "old_function" in str(m.get("content", ""))
            and "deactivated" in str(m.get("content", "")).lower()
            for m in messages_frame.messages
        )
        self.assertTrue(warning_found, "Warning message about deactivated functions not found")

    async def test_no_deactivation_on_reset_strategy(self):
        """Test that functions are not carried over as deactivated on RESET strategy."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
        )
        await flow_manager.initialize()

        # First node with a function
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "old_function",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Old function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)
        self.mock_llm.register_function.reset_mock()

        # Second node without the old function
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "new_function",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "New function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("second", second_node)

        # Verify that old_function was NOT registered in second transition
        registered_function_names = [
            call[0][0] for call in self.mock_llm.register_function.call_args_list
        ]
        self.assertNotIn("old_function", registered_function_names)
        self.assertIn("new_function", registered_function_names)

    async def test_global_functions_not_deactivated(self):
        """Test that global functions are never carried over as deactivated."""
        from pipecat_flows.types import FlowsFunctionSchema

        global_func = FlowsFunctionSchema(
            name="global_function",
            description="A global function",
            properties={},
            required=[],
            handler=AsyncMock(return_value={"status": "success"}),
        )

        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
            global_functions=[global_func],
        )
        await flow_manager.initialize()

        # First node with an additional function
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "node_function",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Node-specific function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)
        self.mock_task.queue_frames.reset_mock()

        # Second node without node_function (but global_function should still be active)
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [],
        }

        await flow_manager._set_node("second", second_node)

        # Verify warning message mentions node_function but NOT global_function
        all_frames = self._get_all_queued_frames()
        messages_frame = next(
            (f for f in all_frames if isinstance(f, LLMMessagesAppendFrame)), None
        )
        self.assertIsNotNone(messages_frame, "LLMMessagesAppendFrame should be queued")

        for msg in messages_frame.messages:
            content = str(msg.get("content", ""))
            if "deactivated" in content.lower():
                self.assertIn("node_function", content)
                self.assertNotIn("global_function", content)

    async def test_deactivated_function_returns_error(self):
        """Test that calling a deactivated function returns an error result."""
        from pipecat_flows.types import FlowsFunctionSchema

        # Create the schema directly to test the helper
        original_schema = FlowsFunctionSchema(
            name="test_function",
            description="Test function",
            properties={"param": {"type": "string"}},
            required=["param"],
            handler=AsyncMock(return_value={"status": "success"}),
        )

        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )

        deactivated_schema = flow_manager._create_deactivated_function_schema(original_schema)

        # Verify schema properties are preserved (with updated description)
        self.assertEqual(deactivated_schema.name, original_schema.name)
        self.assertTrue(deactivated_schema.description.startswith("[DEACTIVATED]"))
        self.assertIn(original_schema.description, deactivated_schema.description)
        self.assertEqual(deactivated_schema.properties, original_schema.properties)
        self.assertEqual(deactivated_schema.required, original_schema.required)

        # Verify deactivated handler returns error (modern signature with flow_manager)
        result = await deactivated_schema.handler({}, flow_manager)
        self.assertEqual(result["status"], "error")
        self.assertIn("deactivated", result["error"])
        self.assertIn("test_function", result["error"])

    async def test_node_level_reset_prevents_deactivation(self):
        """Test that node-level RESET strategy prevents function deactivation."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        # First node with a function
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "old_function",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Old function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)
        self.mock_llm.register_function.reset_mock()

        # Second node with RESET strategy override
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [],
            "context_strategy": ContextStrategyConfig(strategy=ContextStrategy.RESET),
        }

        await flow_manager._set_node("second", second_node)

        # Verify that old_function was NOT registered (no deactivation on RESET)
        registered_function_names = [
            call[0][0] for call in self.mock_llm.register_function.call_args_list
        ]
        self.assertNotIn("old_function", registered_function_names)

    async def test_deactivated_functions_persist_across_multiple_transitions(self):
        """Test that deactivated functions persist through multiple node transitions."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        # First node with function_a
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Function A",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)
        self.mock_task.queue_frames.reset_mock()
        self.mock_llm.register_function.reset_mock()

        # Second node with function_b only (function_a becomes deactivated)
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_b",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Function B",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("second", second_node)

        # Verify function_a was deactivated
        all_frames = self._get_all_queued_frames()
        messages_frame = next(
            (f for f in all_frames if isinstance(f, LLMMessagesAppendFrame)), None
        )
        self.assertIsNotNone(messages_frame, "LLMMessagesAppendFrame should be queued")
        warning_found = any(
            "function_a" in str(msg.get("content", ""))
            and "deactivated" in str(msg.get("content", "")).lower()
            for msg in messages_frame.messages
        )
        self.assertTrue(warning_found, "function_a should be marked as deactivated")

        self.mock_task.queue_frames.reset_mock()
        self.mock_llm.register_function.reset_mock()

        # Third node with function_c only (function_a AND function_b should be deactivated)
        third_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Third task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_c",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Function C",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("third", third_node)

        # Verify both function_a and function_b are mentioned as deactivated
        all_frames = self._get_all_queued_frames()
        messages_frame = next(
            (f for f in all_frames if isinstance(f, LLMMessagesAppendFrame)), None
        )
        self.assertIsNotNone(messages_frame, "LLMMessagesAppendFrame should be queued")

        warning_content = ""
        for msg in messages_frame.messages:
            content = str(msg.get("content", ""))
            if "deactivated" in content.lower():
                warning_content = content
                break

        self.assertIn("function_a", warning_content, "function_a should persist as deactivated")
        self.assertIn("function_b", warning_content, "function_b should now be deactivated")

    async def test_deactivated_prefix_is_idempotent(self):
        """Test that [DEACTIVATED] prefix is not added multiple times."""
        from pipecat_flows.types import FlowsFunctionSchema

        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )

        # Create an already-deactivated schema
        already_deactivated = FlowsFunctionSchema(
            name="test_function",
            description="[DEACTIVATED] Original description",
            properties={},
            required=[],
            handler=AsyncMock(return_value={"status": "error", "error": "deactivated"}),
        )

        # Deactivate it again
        doubly_deactivated = flow_manager._create_deactivated_function_schema(already_deactivated)

        # Verify description only has one [DEACTIVATED] prefix
        self.assertEqual(
            doubly_deactivated.description,
            "[DEACTIVATED] Original description",
            "Should not add multiple [DEACTIVATED] prefixes",
        )
        self.assertEqual(
            doubly_deactivated.description.count("[DEACTIVATED]"),
            1,
            "Should have exactly one [DEACTIVATED] prefix",
        )

    async def test_deactivated_function_reactivated_when_present_in_new_node(self):
        """Test that a previously deactivated function becomes active again when included in new node."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        # Create a handler that we can track
        function_a_handler = AsyncMock(return_value={"status": "success"})

        # First node with function_a
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": function_a_handler,
                        "description": "Function A",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)
        self.mock_task.queue_frames.reset_mock()

        # Second node without function_a (it becomes deactivated)
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [],
        }

        await flow_manager._set_node("second", second_node)

        # Verify function_a is in current_functions as deactivated
        self.assertIn("function_a", flow_manager._current_functions)
        self.assertTrue(
            flow_manager._current_functions["function_a"].description.startswith("[DEACTIVATED]")
        )

        self.mock_task.queue_frames.reset_mock()

        # Third node brings back function_a
        third_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Third task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": function_a_handler,
                        "description": "Function A restored",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("third", third_node)

        # Verify function_a is now active (not deactivated)
        self.assertIn("function_a", flow_manager._current_functions)
        self.assertFalse(
            flow_manager._current_functions["function_a"].description.startswith("[DEACTIVATED]"),
            "function_a should be active again, not deactivated",
        )

        # Verify no deactivation warning was issued for function_a
        all_frames = self._get_all_queued_frames()
        messages_frame = next(
            (f for f in all_frames if isinstance(f, LLMMessagesAppendFrame)), None
        )
        self.assertIsNotNone(messages_frame, "LLMMessagesAppendFrame should be queued")

        for msg in messages_frame.messages:
            content = str(msg.get("content", ""))
            if "deactivated" in content.lower():
                self.assertNotIn(
                    "function_a",
                    content,
                    "function_a should not be in deactivation warning since it's now active",
                )

    async def test_reactivated_function_is_re_registered_with_llm(self):
        """Test that reactivated functions are re-registered with the LLM."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        function_a_handler = AsyncMock(return_value={"status": "success"})

        # First node with function_a
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": function_a_handler,
                        "description": "Function A",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)

        # Count initial registrations for function_a
        initial_registrations = sum(
            1
            for call in self.mock_llm.register_function.call_args_list
            if call[0][0] == "function_a"
        )
        self.assertEqual(initial_registrations, 1, "function_a should be registered once initially")

        # Second node without function_a (it becomes deactivated)
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [],
        }

        await flow_manager._set_node("second", second_node)

        # Count registrations after deactivation
        deactivation_registrations = sum(
            1
            for call in self.mock_llm.register_function.call_args_list
            if call[0][0] == "function_a"
        )
        self.assertEqual(
            deactivation_registrations, 2, "function_a should be re-registered when deactivated"
        )

        # Third node brings back function_a
        third_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Third task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": function_a_handler,
                        "description": "Function A restored",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("third", third_node)

        # Count registrations after reactivation
        final_registrations = sum(
            1
            for call in self.mock_llm.register_function.call_args_list
            if call[0][0] == "function_a"
        )
        self.assertEqual(
            final_registrations,
            3,
            "function_a should be re-registered when reactivated (total 3 registrations)",
        )

    async def test_reactivated_function_handler_works(self):
        """Test that a reactivated function's handler actually works (not the deactivated dummy)."""
        from pipecat.services.llm_service import FunctionCallParams

        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        # Track actual handler calls
        real_handler_called = []

        async def real_handler(args, flow_manager):
            real_handler_called.append(args)
            return {"status": "real_success", "data": args.get("input", "none")}

        # First node with function_a
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": real_handler,
                        "description": "Function A",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)

        # Second node without function_a (deactivates it)
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [],
        }

        await flow_manager._set_node("second", second_node)

        # Third node brings back function_a
        third_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Third task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": real_handler,
                        "description": "Function A restored",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("third", third_node)

        # Get the registered function handler from the most recent registration
        # The last call for function_a should be the reactivated one
        function_a_calls = [
            call
            for call in self.mock_llm.register_function.call_args_list
            if call[0][0] == "function_a"
        ]
        last_registered_handler = function_a_calls[-1][0][1]

        # Create mock params to call the handler
        mock_result_callback = AsyncMock()
        mock_params = MagicMock(spec=FunctionCallParams)
        mock_params.arguments = {"input": "test_value"}
        mock_params.result_callback = mock_result_callback

        # Call the registered handler
        await last_registered_handler(mock_params)

        # Verify the real handler was called, not the deactivated dummy
        self.assertEqual(len(real_handler_called), 1, "Real handler should have been called")
        self.assertEqual(real_handler_called[0], {"input": "test_value"})

        # Verify result callback was called with success (not error)
        mock_result_callback.assert_called_once()
        result = mock_result_callback.call_args[0][0]
        self.assertEqual(result.get("status"), "real_success")
        self.assertNotIn("error", result)

    async def test_reactivated_function_description_no_deactivated_prefix(self):
        """Test that reactivated function's stored schema has no [DEACTIVATED] prefix."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        handler = AsyncMock(return_value={"status": "success"})

        # First node with function_a
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": handler,
                        "description": "Original description",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)
        self.assertEqual(
            flow_manager._current_functions["function_a"].description, "Original description"
        )

        # Second node without function_a (deactivates it)
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [],
        }

        await flow_manager._set_node("second", second_node)
        self.assertTrue(
            flow_manager._current_functions["function_a"].description.startswith("[DEACTIVATED]")
        )

        # Third node reactivates function_a
        third_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Third task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": handler,
                        "description": "Restored description",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("third", third_node)

        # Verify stored schema has clean description
        self.assertEqual(
            flow_manager._current_functions["function_a"].description,
            "Restored description",
            "Reactivated function should have clean description without [DEACTIVATED] prefix",
        )
        self.assertFalse(
            flow_manager._current_functions["function_a"].description.startswith("[DEACTIVATED]")
        )

    async def test_functions_always_registered_on_each_transition(self):
        """Test that functions are always re-registered on each node transition."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        handler = AsyncMock(return_value={"status": "success"})

        # Node with function_a and function_b
        node_with_both: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": handler,
                        "description": "Function A",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "function_b",
                        "handler": handler,
                        "description": "Function B",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        # First transition
        await flow_manager._set_node("first", node_with_both)

        first_transition_registrations = {
            call[0][0] for call in self.mock_llm.register_function.call_args_list
        }
        self.assertIn("function_a", first_transition_registrations)
        self.assertIn("function_b", first_transition_registrations)

        initial_call_count = len(self.mock_llm.register_function.call_args_list)

        # Second transition with same functions
        await flow_manager._set_node("second", node_with_both)

        # Verify functions were registered again
        new_call_count = len(self.mock_llm.register_function.call_args_list)
        new_registrations = new_call_count - initial_call_count

        self.assertGreaterEqual(
            new_registrations, 2, "Both functions should be re-registered on second transition"
        )

    async def test_deactivated_warnings_transform_frame_queued(self):
        """Test that an LLMMessagesTransformFrame is queued to remove old warnings."""
        from pipecat.frames.frames import LLMMessagesTransformFrame

        from pipecat_flows.manager import FlowManager

        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        handler = AsyncMock(return_value={"status": "success"})

        # First node with function_a
        first_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_a",
                        "handler": handler,
                        "description": "Function A",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("first", first_node)
        self.mock_task.queue_frames.reset_mock()

        # Second node with only function_b (function_a becomes deactivated)
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "function_b",
                        "handler": handler,
                        "description": "Function B",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }

        await flow_manager._set_node("second", second_node)

        # Verify LLMMessagesTransformFrame was queued
        all_frames = []
        for call in self.mock_task.queue_frames.call_args_list:
            frames = call[0][0]
            all_frames.extend(frames)

        transform_frames = [f for f in all_frames if isinstance(f, LLMMessagesTransformFrame)]
        self.assertGreater(
            len(transform_frames),
            0,
            "LLMMessagesTransformFrame should be queued to remove old warnings",
        )

    async def test_deactivated_warnings_transform_function_removes_warnings(self):
        """Test that the transform function correctly removes deactivated warning messages."""
        from pipecat_flows.manager import FlowManager

        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        warning_marker = FlowManager._DEACTIVATED_FUNCTIONS_MESSAGE_PREFIX

        # Test messages with warnings mixed in
        input_messages = [
            {"role": "system", "content": f"{warning_marker}Old warning 1"},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": f"{warning_marker}Old warning 2"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "system", "content": f"{warning_marker}Old warning 3"},
            {"role": "system", "content": "Regular system message"},
        ]

        # Apply the transform method directly
        result = flow_manager._remove_deactivated_functions_messages(input_messages)

        # Verify all warnings were removed
        warning_count = sum(
            1
            for msg in result
            if isinstance(msg.get("content", ""), str) and msg["content"].startswith(warning_marker)
        )
        self.assertEqual(warning_count, 0, "All warnings should be removed")

        # Verify non-warning messages are preserved
        self.assertEqual(len(result), 3, "Should have 3 messages remaining")
        user_messages = [msg for msg in result if msg.get("role") == "user"]
        assistant_messages = [msg for msg in result if msg.get("role") == "assistant"]
        system_messages = [msg for msg in result if msg.get("role") == "system"]
        self.assertEqual(len(user_messages), 1, "User message should be preserved")
        self.assertEqual(len(assistant_messages), 1, "Assistant message should be preserved")
        self.assertEqual(len(system_messages), 1, "Non-warning system message should be preserved")
        self.assertEqual(
            system_messages[0]["content"],
            "Regular system message",
            "Regular system message content should be preserved",
        )
