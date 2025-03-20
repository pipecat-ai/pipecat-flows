#
# Copyright (c) 2024, Daily
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
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.google import GoogleLLMService
from pipecat.services.openai import OpenAILLMService

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

        # Set up mock LLM with client
        self.mock_llm = OpenAILLMService(api_key="")
        self.mock_llm._client = MagicMock()
        self.mock_llm._client.chat = MagicMock()
        self.mock_llm._client.chat.completions = MagicMock()
        self.mock_llm._client.chat.completions.create = AsyncMock()

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
        self.mock_context_aggregator.user.return_value.get_context_frame = MagicMock(
            return_value=MagicMock()
        )

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

    async def test_default_strategy(self):
        """Test default context strategy (APPEND)."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # First node should use UpdateFrame regardless of strategy
        await flow_manager.set_node("first", self.sample_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in first_frames))

        # Reset mock
        self.mock_task.queue_frames.reset_mock()

        # Subsequent node should use AppendFrame with default strategy
        await flow_manager.set_node("second", self.sample_node)
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
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
        await flow_manager.set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        # Second node should use UpdateFrame with RESET strategy
        await flow_manager.set_node("second", self.sample_node)
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames))

    async def test_reset_with_summary_success(self):
        """Test successful RESET_WITH_SUMMARY strategy."""
        # Mock successful summary generation
        mock_summary = "Conversation summary"
        self.mock_llm._client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=mock_summary))
        ]

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
        await flow_manager.set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        await flow_manager.set_node("second", self.sample_node)

        # Verify summary was included in context update
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        update_frame = next(f for f in second_frames if isinstance(f, LLMMessagesUpdateFrame))
        self.assertTrue(any(mock_summary in str(m) for m in update_frame.messages))

    async def test_reset_with_summary_timeout(self):
        """Test RESET_WITH_SUMMARY fallback on timeout."""
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
        self.mock_llm._client.chat.completions.create.side_effect = AsyncMock(
            side_effect=TimeoutError
        )

        # Set nodes and verify fallback to RESET
        await flow_manager.set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        await flow_manager.set_node("second", self.sample_node)

        # Verify UpdateFrame was used (RESET behavior)
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames))

    async def test_provider_specific_summary_formatting(self):
        """Test summary formatting for different LLM providers."""
        summary = "Test summary"

        # Test OpenAI format
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=OpenAILLMService(api_key=""),
            context_aggregator=self.mock_context_aggregator,
        )
        openai_message = flow_manager.adapter.format_summary_message(summary)
        self.assertEqual(openai_message["role"], "system")

        # Test Anthropic format
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=AnthropicLLMService(api_key=""),
            context_aggregator=self.mock_context_aggregator,
        )
        anthropic_message = flow_manager.adapter.format_summary_message(summary)
        self.assertEqual(anthropic_message["role"], "user")

        # Test Gemini format
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=GoogleLLMService(api_key=""),
            context_aggregator=self.mock_context_aggregator,
        )
        gemini_message = flow_manager.adapter.format_summary_message(summary)
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
        await flow_manager.set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        await flow_manager.set_node("second", node_with_strategy)

        # Verify UpdateFrame was used (RESET behavior) despite global APPEND
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames))

    async def test_summary_generation_content(self):
        """Test that summary generation uses correct prompt and context."""
        mock_summary = "Generated summary"
        self.mock_llm._client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=mock_summary))
        ]

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
        await flow_manager.set_node("first", self.sample_node)
        await flow_manager.set_node("second", self.sample_node)

        # Verify summary generation call
        create_call = self.mock_llm._client.chat.completions.create.call_args
        create_kwargs = create_call[1]

        # Verify prompt and context were included
        messages = create_kwargs["messages"]
        self.assertTrue(any(summary_prompt in str(m) for m in messages))
        self.assertTrue(
            any(str(self.mock_context.messages[0]["content"]) in str(m) for m in messages)
        )

    async def test_context_structure_after_summary(self):
        """Test the structure of context after summary generation."""
        mock_summary = "Generated summary"
        self.mock_llm._client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=mock_summary))
        ]

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
        await flow_manager.set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        # Node with new task messages
        new_node = {
            "task_messages": [{"role": "system", "content": "New task."}],
            "functions": [],
        }
        await flow_manager.set_node("second", new_node)

        # Verify context structure
        update_call = self.mock_task.queue_frames.call_args_list[0]
        update_frames = update_call[0][0]
        messages_frame = next(f for f in update_frames if isinstance(f, LLMMessagesUpdateFrame))

        # Verify order: summary message, then new task messages
        self.assertTrue(mock_summary in str(messages_frame.messages[0]))
        self.assertEqual(
            messages_frame.messages[1]["content"], new_node["task_messages"][0]["content"]
        )
