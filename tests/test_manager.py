#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Test suite for FlowManager functionality.

This module contains tests for the FlowManager class, which handles conversation
flow management across different LLM providers. Tests cover:
- Static and dynamic flow initialization
- State transitions and validation
- Function registration and execution
- Action handling
- Error cases

The tests use unittest.IsolatedAsyncioTestCase for async support and
include mocked dependencies for PipelineTask, LLM services, and TTS.
"""

import unittest
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from pipecat.frames.frames import LLMMessagesAppendFrame, LLMMessagesUpdateFrame
from pipecat.services.openai import OpenAILLMService

from pipecat_flows.exceptions import FlowError, FlowInitializationError, FlowTransitionError
from pipecat_flows.manager import FlowConfig, FlowManager, NodeConfig
from pipecat_flows.types import FlowArgs, FlowResult


class TestFlowManager(unittest.IsolatedAsyncioTestCase):
    """Test suite for FlowManager class.

    Tests functionality of FlowManager including:
    - Static and dynamic flow initialization
    - State transitions
    - Function registration
    - Action execution
    - Error handling
    - Node validation
    """

    async def asyncSetUp(self):
        """Set up test fixtures before each test."""
        self.mock_task = AsyncMock()
        self.mock_task.event_handler = Mock()
        self.mock_llm = OpenAILLMService(api_key="")
        self.mock_llm.register_function = MagicMock()
        self.mock_tts = AsyncMock()

        # Create mock context aggregator
        self.mock_context_aggregator = MagicMock()
        self.mock_context_aggregator.user = MagicMock()
        self.mock_context_aggregator.user.return_value = MagicMock()
        self.mock_context_aggregator.user.return_value.get_context_frame = MagicMock(
            return_value=MagicMock()
        )

        self.mock_result_callback = AsyncMock()

        # Sample node configurations
        self.sample_node: NodeConfig = {
            "role_messages": [{"role": "system", "content": "You are a helpful test assistant."}],
            "task_messages": [{"role": "system", "content": "Complete the test task."}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "description": "Test function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        # Sample static flow configuration
        self.static_flow_config = {
            "initial_node": "start",
            "nodes": {
                "start": self.sample_node,
                "next_node": self.sample_node,
            },
        }

    async def test_static_flow_initialization(self):
        """Test initialization of a static flow configuration."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            tts=self.mock_tts,
            flow_config=FlowConfig(**self.static_flow_config),
        )

        # Verify static mode setup
        self.assertEqual(flow_manager.initial_node, "start")
        self.assertEqual(flow_manager.nodes, self.static_flow_config["nodes"])
        # No need to check transition_callbacks anymore as they're now inline

    async def test_dynamic_flow_initialization(self):
        """Test initialization of dynamic flow."""
        # Create mock transition callback
        mock_transition_handler = AsyncMock()

        # Initialize flow manager
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            tts=self.mock_tts,
        )

        # Create test node with transition callback
        test_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test message"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "description": "Test function",
                        "parameters": {},
                        "transition_callback": mock_transition_handler,
                    },
                }
            ],
        }

        # Initialize and set node
        await flow_manager.initialize()
        await flow_manager.set_node("test", test_node)

        self.assertFalse(
            mock_transition_handler.called
        )  # Shouldn't be called until function is used
        self.assertEqual(flow_manager.current_node, "test")

    async def test_static_flow_transitions(self):
        """Test transitions in static flows."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            tts=self.mock_tts,
            flow_config=FlowConfig(**self.static_flow_config),
        )

        # Initialize and transition to first node
        await flow_manager.initialize()
        self.assertEqual(flow_manager.current_node, "start")

        # Clear mock call history to focus on transition
        self.mock_task.queue_frames.reset_mock()

        # Test transition to next node
        await flow_manager._handle_static_transition("next_node", {}, flow_manager)

        # Verify node transition occurred
        self.assertEqual(flow_manager.current_node, "next_node")

        # Verify frame handling
        self.assertEqual(self.mock_task.queue_frames.call_count, 2)

        # Get the first call (context update)
        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]

        # Should have exactly one AppendFrame
        append_frames = [f for f in first_frames if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(len(append_frames), 1, "Should have exactly one AppendFrame")

    async def test_transition_callback_signatures(self):
        """Test both two and three argument transition callback signatures."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            tts=self.mock_tts,
        )
        await flow_manager.initialize()

        # Track callback executions
        old_style_called = False
        new_style_called = False
        received_result = None

        # Old style callback (args, flow_manager)
        async def old_style_callback(args: Dict, flow_manager: FlowManager):
            nonlocal old_style_called
            old_style_called = True

        # New style callback (args, result, flow_manager)
        async def new_style_callback(args: Dict, result: FlowResult, flow_manager: FlowManager):
            nonlocal new_style_called, received_result
            new_style_called = True
            received_result = result

        # Test handler that returns a known result
        async def test_handler(args: FlowArgs) -> FlowResult:
            return {"status": "success", "test_data": "test_value"}

        # Create and test old style node
        old_style_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "old_style_function",
                        "handler": test_handler,
                        "description": "Test function",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_callback": old_style_callback,
                    },
                }
            ],
        }

        # Test old style callback
        await flow_manager.set_node("old_style", old_style_node)
        func = flow_manager.llm.register_function.call_args[0][1]

        # Store the context_updated callback
        context_updated_callback = None

        async def result_callback(result, properties=None):
            nonlocal context_updated_callback
            if properties and properties.on_context_updated:
                context_updated_callback = properties.on_context_updated

        # Call function and get context_updated callback
        await func("old_style_function", "id", {}, None, None, result_callback)

        # Execute the context_updated callback
        self.assertIsNotNone(context_updated_callback, "Context updated callback not set")
        await context_updated_callback()

        self.assertTrue(old_style_called, "Old style callback was not called")

        # Reset and test new style node
        flow_manager.llm.register_function.reset_mock()
        new_style_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "new_style_function",
                        "handler": test_handler,
                        "description": "Test function",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_callback": new_style_callback,
                    },
                }
            ],
        }

        # Test new style callback
        await flow_manager.set_node("new_style", new_style_node)
        func = flow_manager.llm.register_function.call_args[0][1]

        # Reset context_updated callback
        context_updated_callback = None

        # Call function and get context_updated callback
        await func("new_style_function", "id", {}, None, None, result_callback)

        # Execute the context_updated callback
        self.assertIsNotNone(context_updated_callback, "Context updated callback not set")
        await context_updated_callback()

        self.assertTrue(new_style_called, "New style callback was not called")
        self.assertIsNotNone(received_result, "Result was not passed to callback")
        self.assertEqual(received_result["test_data"], "test_value")

    async def test_node_validation(self):
        """Test node configuration validation."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Test missing task_messages
        invalid_config = {"functions": []}
        with self.assertRaises(FlowError) as context:
            await flow_manager.set_node("test", invalid_config)
        self.assertIn("missing required 'task_messages' field", str(context.exception))

        # Test missing functions
        invalid_config = {"task_messages": []}
        with self.assertRaises(FlowError) as context:
            await flow_manager.set_node("test", invalid_config)
        self.assertIn("missing required 'functions' field", str(context.exception))

    async def test_function_registration(self):
        """Test function registration with LLM."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Reset mock to clear initialization calls
        self.mock_llm.register_function.reset_mock()

        # Set node with function
        await flow_manager.set_node("test", self.sample_node)

        # Verify function was registered
        self.mock_llm.register_function.assert_called_once()
        name, func = self.mock_llm.register_function.call_args[0]
        self.assertEqual(name, "test_function")
        self.assertTrue(callable(func))

    async def test_action_execution(self):
        """Test execution of pre and post actions."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            tts=self.mock_tts,
        )
        await flow_manager.initialize()

        # Create node config with actions
        node_with_actions: NodeConfig = {
            "role_messages": self.sample_node["role_messages"],
            "task_messages": self.sample_node["task_messages"],
            "functions": self.sample_node["functions"],
            "pre_actions": [{"type": "tts_say", "text": "Pre action"}],
            "post_actions": [{"type": "tts_say", "text": "Post action"}],
        }

        # Reset mock to clear initialization calls
        self.mock_tts.say.reset_mock()

        # Set node with actions
        await flow_manager.set_node("test", node_with_actions)

        # Verify TTS was called for both actions
        self.assertEqual(self.mock_tts.say.call_count, 2)
        self.mock_tts.say.assert_any_call("Pre action")
        self.mock_tts.say.assert_any_call("Post action")

    async def test_error_handling(self):
        """Test error handling in flow manager.

        Verifies:
        1. Cannot set node before initialization
        2. Initialization fails properly when task queue fails
        3. Node setting fails when task queue fails
        """
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )

        # Test setting node before initialization
        with self.assertRaises(FlowTransitionError):
            await flow_manager.set_node("test", self.sample_node)

        # Initialize normally
        await flow_manager.initialize()
        self.assertTrue(flow_manager.initialized)

        # Test node setting error
        self.mock_task.queue_frames.side_effect = Exception("Queue error")
        with self.assertRaises(FlowError):
            await flow_manager.set_node("test", self.sample_node)

        # Verify flow manager remains initialized despite error
        self.assertTrue(flow_manager.initialized)

    async def test_state_management(self):
        """Test state management across nodes."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Set state data
        test_value = "test_value"
        flow_manager.state["test_key"] = test_value

        # Reset mock to clear initialization calls
        self.mock_task.queue_frames.reset_mock()

        # Verify state persists across node transitions
        await flow_manager.set_node("test", self.sample_node)
        self.assertEqual(flow_manager.state["test_key"], test_value)

    async def test_multiple_function_registration(self):
        """Test registration of multiple functions."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create node config with multiple functions
        node_config: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": f"func_{i}",
                        "handler": AsyncMock(return_value={"status": "success"}),
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for i in range(3)
            ],
        }

        await flow_manager.set_node("test", node_config)

        # Verify all functions were registered
        self.assertEqual(self.mock_llm.register_function.call_count, 3)
        self.assertEqual(len(flow_manager.current_functions), 3)

    async def test_initialize_already_initialized(self):
        """Test initializing an already initialized flow manager."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Try to initialize again
        with patch("loguru.logger.warning") as mock_logger:
            await flow_manager.initialize()
            mock_logger.assert_called_once()

    async def test_register_action(self):
        """Test registering custom actions."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )

        async def custom_action(action):
            pass

        flow_manager.register_action("custom", custom_action)
        self.assertIn("custom", flow_manager.action_manager.action_handlers)

    async def test_call_handler_variations(self):
        """Test different handler signature variations."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Test handler with args
        async def handler_with_args(args):
            return {"status": "success", "args": args}

        result = await flow_manager._call_handler(handler_with_args, {"test": "value"})
        self.assertEqual(result["args"]["test"], "value")

        # Test handler without args
        async def handler_no_args():
            return {"status": "success"}

        result = await flow_manager._call_handler(handler_no_args, {})
        self.assertEqual(result["status"], "success")

    async def test_transition_func_error_handling(self):
        """Test error handling in transition functions."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        async def error_handler(args):
            raise ValueError("Test error")

        transition_func = await flow_manager._create_transition_func(
            "test", error_handler, transition_to=None
        )

        # Mock result callback
        callback_called = False

        async def result_callback(result):
            nonlocal callback_called
            callback_called = True
            self.assertIn("error", result)
            self.assertEqual(result["status"], "error")
            self.assertIn("Test error", result["error"])

        # The transition function should catch the error and pass it to the callback
        await transition_func("test", "id", {}, None, None, result_callback)
        self.assertTrue(callback_called, "Result callback was not called")

    async def test_node_validation_edge_cases(self):
        """Test edge cases in node validation."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Test function with missing name
        invalid_config = {
            "task_messages": [{"role": "system", "content": "Test"}],
            "functions": [{"type": "function"}],  # Missing name
        }
        with self.assertRaises(FlowError) as context:
            await flow_manager.set_node("test", invalid_config)
        self.assertIn("missing name field", str(context.exception))

        # Test node function without handler or transition_to
        invalid_config = {
            "task_messages": [{"role": "system", "content": "Test"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_func",
                        "description": "Test",
                        "parameters": {},
                    },
                }
            ],
        }

        # Mock loguru.logger.warning to capture the warning
        warning_message = None

        def capture_warning(msg, *args, **kwargs):
            nonlocal warning_message
            warning_message = msg

        with patch("loguru.logger.warning", side_effect=capture_warning):
            await flow_manager.set_node("test", invalid_config)
            self.assertIsNotNone(warning_message)
            self.assertIn(
                "Function 'test_func' in node 'test' has neither handler, transition_to, nor transition_callback",
                warning_message,
            )

    async def test_transition_callback_error_handling(self):
        """Test error handling in transition callback."""

        async def failing_handler(args, flow_manager):
            raise ValueError("Transition error")

        # Initialize flow manager
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            tts=self.mock_tts,
        )
        await flow_manager.initialize()

        # Create test node with failing transition callback
        test_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test message"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "description": "Test function",
                        "parameters": {},
                        "transition_callback": failing_handler,
                    },
                }
            ],
        }

        # Set up node and get registered function
        await flow_manager.set_node("test", test_node)
        transition_func = flow_manager.llm.register_function.call_args[0][1]

        # Track the result and context_updated callback
        context_updated_callback = None
        final_results = []

        async def mock_result_callback(result, properties=None):
            nonlocal context_updated_callback
            final_results.append(result)
            if properties and properties.on_context_updated:
                context_updated_callback = properties.on_context_updated

        # Call function
        await transition_func(
            "test_function",
            "test_id",
            {},
            self.mock_llm,
            {},
            mock_result_callback,
        )

        # Execute the context updated callback which should trigger the error
        self.assertIsNotNone(context_updated_callback, "Context updated callback not set")
        try:
            await context_updated_callback()
        except ValueError:
            pass  # Expected error

        # Verify error handling - should have two results:
        # 1. The initial acknowledged status
        # 2. The error status after the callback fails
        self.assertEqual(len(final_results), 2)
        self.assertEqual(final_results[0]["status"], "acknowledged")
        self.assertEqual(final_results[1]["status"], "error")
        self.assertIn("Transition error", final_results[1]["error"])

    async def test_register_function_error_handling(self):
        """Test error handling in function registration."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Mock LLM to raise error on register_function
        flow_manager.llm.register_function.side_effect = Exception("Registration error")

        new_functions = set()
        with self.assertRaises(FlowError):
            await flow_manager._register_function("test", None, None, new_functions)

    async def test_action_execution_error_handling(self):
        """Test error handling in action execution."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create node config with actions that will fail
        node_config: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test"}],
            "functions": [],
            "pre_actions": [{"type": "invalid_action"}],
            "post_actions": [{"type": "another_invalid_action"}],
        }

        # Should raise FlowError due to invalid actions
        with self.assertRaises(FlowError):
            await flow_manager.set_node("test", node_config)

        # Verify error handling for pre and post actions separately
        with self.assertRaises(FlowError):
            await flow_manager._execute_actions(pre_actions=[{"type": "invalid_action"}])

        with self.assertRaises(FlowError):
            await flow_manager._execute_actions(post_actions=[{"type": "invalid_action"}])

    async def test_update_llm_context_error_handling(self):
        """Test error handling in LLM context updates."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Mock task to raise error on queue_frames
        flow_manager.task.queue_frames.side_effect = Exception("Queue error")

        with self.assertRaises(FlowError):
            await flow_manager._update_llm_context(
                messages=[{"role": "system", "content": "Test"}], functions=[]
            )

    async def test_handler_removal_all_formats(self):
        """Test handler removal from different function configurations."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        async def dummy_handler(args):
            return {"status": "success"}

        # Test OpenAI format
        openai_config = {
            "type": "function",
            "function": {"name": "test", "handler": dummy_handler, "parameters": {}},
        }
        flow_manager._remove_handlers(openai_config)
        self.assertNotIn("handler", openai_config["function"])

        # Test Anthropic format
        anthropic_config = {"name": "test", "handler": dummy_handler, "parameters": {}}
        flow_manager._remove_handlers(anthropic_config)
        self.assertNotIn("handler", anthropic_config)

        # Test Gemini format
        gemini_config = {
            "function_declarations": [
                {"name": "test1", "handler": dummy_handler, "parameters": {}},
                {"name": "test2", "handler": dummy_handler, "parameters": {}},
            ]
        }
        flow_manager._remove_handlers(gemini_config)
        for decl in gemini_config["function_declarations"]:
            self.assertNotIn("handler", decl)

    async def test_function_declarations_processing(self):
        """Test processing of function declarations format."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        async def test_handler(args):
            return {"status": "success"}

        # Create node config with OpenAI format for multiple functions
        node_config: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "test1",
                        "handler": test_handler,
                        "description": "Test function 1",
                        "parameters": {},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "test2",
                        "handler": test_handler,
                        "description": "Test function 2",
                        "parameters": {},
                    },
                },
            ],
        }

        # Set node and verify function registration
        await flow_manager.set_node("test", node_config)

        # Verify both functions were registered
        self.assertIn("test1", flow_manager.current_functions)
        self.assertIn("test2", flow_manager.current_functions)

    async def test_function_token_handling_main_module(self):
        """Test handling of __function__: tokens when function is in main module."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Define test handler in main module
        async def test_handler_main(args):
            return {"status": "success"}

        # Add handler to main module
        import sys

        sys.modules["__main__"].test_handler_main = test_handler_main

        try:
            node_config: NodeConfig = {
                "task_messages": [{"role": "system", "content": "Test"}],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "test_function",
                            "handler": "__function__:test_handler_main",
                            "description": "Test function",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }

            await flow_manager.set_node("test", node_config)
            self.assertIn("test_function", flow_manager.current_functions)

        finally:
            # Clean up
            delattr(sys.modules["__main__"], "test_handler_main")

    async def test_function_token_handling_not_found(self):
        """Test error handling when function is not found in any module."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        node_config: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "handler": "__function__:nonexistent_handler",
                        "description": "Test function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        with self.assertRaises(FlowError) as context:
            await flow_manager.set_node("test", node_config)

        self.assertIn("Function 'nonexistent_handler' not found", str(context.exception))

    async def test_function_token_execution(self):
        """Test that functions registered with __function__: token work when called."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Define and register test handler
        test_called = False

        async def test_handler(args):
            nonlocal test_called
            test_called = True
            return {"status": "success", "args": args}

        import sys

        sys.modules["__main__"].test_handler = test_handler

        try:
            node_config: NodeConfig = {
                "task_messages": [{"role": "system", "content": "Test"}],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "test_function",
                            "handler": "__function__:test_handler",
                            "description": "Test function",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }

            await flow_manager.set_node("test", node_config)

            # Get the registered function and test it
            name, func = self.mock_llm.register_function.call_args[0]

            async def callback(result, properties=None):
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["args"], {"test": "value"})

            await func("test_function", "id", {"test": "value"}, None, None, callback)
            self.assertTrue(test_called)

        finally:
            delattr(sys.modules["__main__"], "test_handler")

    async def test_role_message_inheritance(self):
        """Test that role messages are properly handled between nodes."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # First node with role messages
        first_node: NodeConfig = {
            "role_messages": [{"role": "system", "content": "You are a helpful assistant."}],
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [],
        }

        # Second node without role messages
        second_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [],
        }

        # Set first node and verify UpdateFrame
        await flow_manager.set_node("first", first_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]  # Get first call
        first_frames = first_call[0][0]
        update_frames = [f for f in first_frames if isinstance(f, LLMMessagesUpdateFrame)]
        self.assertEqual(len(update_frames), 1)

        # Verify combined messages in first node
        expected_first_messages = first_node["role_messages"] + first_node["task_messages"]
        self.assertEqual(update_frames[0].messages, expected_first_messages)

        # Reset mock and set second node
        self.mock_task.queue_frames.reset_mock()
        await flow_manager.set_node("second", second_node)

        # Verify AppendFrame for second node
        first_call = self.mock_task.queue_frames.call_args_list[0]  # Get first call
        second_frames = first_call[0][0]
        append_frames = [f for f in second_frames if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(len(append_frames), 1)

        # Verify only task messages in second node
        self.assertEqual(append_frames[0].messages, second_node["task_messages"])

    async def test_frame_type_selection(self):
        """Test that correct frame types are used based on node order."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        test_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test task."}],
            "functions": [],
        }

        # First node should use UpdateFrame
        await flow_manager.set_node("first", test_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]  # Get first call
        first_frames = first_call[0][0]
        self.assertTrue(
            any(isinstance(f, LLMMessagesUpdateFrame) for f in first_frames),
            "First node should use UpdateFrame",
        )
        self.assertFalse(
            any(isinstance(f, LLMMessagesAppendFrame) for f in first_frames),
            "First node should not use AppendFrame",
        )

        # Reset mock
        self.mock_task.queue_frames.reset_mock()

        # Second node should use AppendFrame
        await flow_manager.set_node("second", test_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]  # Get first call
        second_frames = first_call[0][0]
        self.assertTrue(
            any(isinstance(f, LLMMessagesAppendFrame) for f in second_frames),
            "Subsequent nodes should use AppendFrame",
        )
        self.assertFalse(
            any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames),
            "Subsequent nodes should not use UpdateFrame",
        )

    async def test_edge_vs_node_function_behavior(self):
        """Test different completion behavior for edge and node functions."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create test functions
        async def test_handler(args):
            return {"status": "success"}

        # Create node with both types of functions
        node_config: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "node_function",
                        "handler": test_handler,
                        "description": "Node function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "edge_function",
                        "handler": test_handler,
                        "description": "Edge function",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "next_node",
                    },
                },
            ],
        }

        await flow_manager.set_node("test", node_config)

        # Get the registered functions
        node_func = None
        edge_func = None
        for args in self.mock_llm.register_function.call_args_list:
            name = args[0][0]
            func = args[0][1]
            if name == "node_function":
                node_func = func
            elif name == "edge_function":
                edge_func = func

        # Test node function
        self.mock_task.queue_frames.reset_mock()
        node_result = None
        node_properties = None

        async def node_callback(result, *, properties=None):
            nonlocal node_result, node_properties
            node_result = result
            node_properties = properties

        await node_func("node_function", "id1", {}, None, None, node_callback)
        # Node function should not set run_llm=False
        self.assertTrue(node_properties is None or node_properties.run_llm is not False)

        # Test edge function
        self.mock_task.queue_frames.reset_mock()
        edge_result = None
        edge_properties = None

        async def edge_callback(result, *, properties=None):
            nonlocal edge_result, edge_properties
            edge_result = result
            edge_properties = properties

        await edge_func("edge_function", "id2", {}, None, None, edge_callback)
        # Edge function should set run_llm=False
        self.assertTrue(edge_properties is not None and edge_properties.run_llm is False)

    async def test_completion_timing(self):
        """Test that completions occur at the right time."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Test initial node setup
        self.mock_task.queue_frames.reset_mock()
        await flow_manager.set_node(
            "initial",
            {
                "task_messages": [{"role": "system", "content": "Test"}],
                "functions": [],
            },
        )
        # Should see two calls: context update and completion trigger
        self.assertEqual(self.mock_task.queue_frames.call_count, 2)

        # Add next node to flow manager's nodes
        next_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Next test"}],
            "functions": [],
        }
        flow_manager.nodes["next"] = next_node

        # Test node transition
        self.mock_task.queue_frames.reset_mock()
        await flow_manager._handle_static_transition("next", {}, flow_manager)
        # Should see two calls: context update and completion trigger
        self.assertEqual(self.mock_task.queue_frames.call_count, 2)

    async def test_transition_configuration_exclusivity(self):
        """Test that transition_to and transition_callback cannot be used together."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Create mock transition callback
        mock_transition_handler = AsyncMock()

        # Create test node with both transition types
        test_node: NodeConfig = {
            "task_messages": [{"role": "system", "content": "Test message"}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "description": "Test function",
                        "parameters": {},
                        "transition_to": "next_node",
                        "transition_callback": mock_transition_handler,
                    },
                }
            ],
        }

        # Should raise error when trying to use both
        with self.assertRaises(FlowError) as context:
            await flow_manager.set_node("test", test_node)
        self.assertIn(
            "cannot have both transition_to and transition_callback", str(context.exception)
        )

    async def test_get_current_context(self):
        """Test getting current conversation context."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Mock the context messages
        mock_messages = [{"role": "system", "content": "Test message"}]
        self.mock_context_aggregator.user()._context.messages = mock_messages

        # Test getting context
        context = flow_manager.get_current_context()
        self.assertEqual(context, mock_messages)

        # Test error when context aggregator is not available
        flow_manager._context_aggregator = None
        with self.assertRaises(FlowError) as context:
            flow_manager.get_current_context()
        self.assertIn("No context aggregator available", str(context.exception))

    async def test_handler_with_flow_manager(self):
        """Test function handler that receives both args and flow_manager."""
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        handler_called = False
        correct_flow_manager = False

        async def modern_handler(args: FlowArgs, flow_mgr: FlowManager) -> FlowResult:
            nonlocal handler_called, correct_flow_manager
            handler_called = True
            correct_flow_manager = flow_mgr is flow_manager
            return {"status": "success", "args_received": args, "has_flow_manager": True}

        result = await flow_manager._call_handler(modern_handler, {"test": "value"})

        self.assertTrue(handler_called)
        self.assertTrue(correct_flow_manager)
        self.assertEqual(result["args_received"]["test"], "value")
        self.assertTrue(result["has_flow_manager"])
