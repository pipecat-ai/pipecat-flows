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
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import LLMMessagesAppendFrame, LLMMessagesUpdateFrame
from pipecat.services.openai import OpenAILLMService

from pipecat_flows.exceptions import FlowError, FlowInitializationError, FlowTransitionError
from pipecat_flows.manager import FlowManager


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
        self.mock_llm = MagicMock(spec=OpenAILLMService)
        self.mock_tts = AsyncMock()

        # Sample node configurations
        self.sample_node = {
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
            tts=self.mock_tts,
            flow_config=self.static_flow_config,
        )

        # Verify static mode setup
        self.assertEqual(flow_manager.initial_node, "start")
        self.assertEqual(flow_manager.nodes, self.static_flow_config["nodes"])
        self.assertEqual(flow_manager.transition_callback.__name__, "_handle_static_transition")

        # Initialize flow
        await flow_manager.initialize()

        # Verify initialization
        self.assertTrue(flow_manager.initialized)

        # Verify the initial node was set
        self.assertEqual(flow_manager.current_node, "start")

        # Verify the messages were queued with UpdateFrame
        calls = self.mock_task.queue_frames.call_args_list
        self.assertEqual(len(calls), 1)  # Should be called once

        # Get the frames from the first call
        frames = calls[0][0][0]  # First call, first argument, which is the list of frames
        update_frames = [f for f in frames if isinstance(f, LLMMessagesUpdateFrame)]
        self.assertEqual(len(update_frames), 1)

        # Verify the combined messages were sent
        expected_messages = self.sample_node["role_messages"] + self.sample_node["task_messages"]
        actual_messages = update_frames[0].messages
        self.assertEqual(actual_messages, expected_messages)

    async def test_dynamic_flow_initialization(self):
        """Test initialization of dynamic flow."""
        # Create mock transition callback
        mock_transition_callback = AsyncMock()

        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            tts=self.mock_tts,
            transition_callback=mock_transition_callback,
        )

        # Verify dynamic mode setup
        self.assertIsNone(flow_manager.initial_node)
        self.assertEqual(flow_manager.nodes, {})
        self.assertEqual(flow_manager.transition_callback, mock_transition_callback)

        # Initialize flow
        await flow_manager.initialize()

        # Verify initialization
        self.assertTrue(flow_manager.initialized)

        # Verify no messages were queued during initialization
        self.mock_task.queue_frames.assert_not_called()

        # Create and set initial node
        initial_node = {
            "role_messages": [{"role": "system", "content": "You are a helpful assistant."}],
            "task_messages": [{"role": "system", "content": "Ask the user for their name."}],
            "functions": [],
        }

        # Reset mock to clear any previous calls
        self.mock_task.queue_frames.reset_mock()

        # Set initial node
        await flow_manager.set_node("initial", initial_node)

        # Verify frames were queued
        self.mock_task.queue_frames.assert_called_once()
        frames = self.mock_task.queue_frames.call_args[0][0]

        # Should have exactly one UpdateFrame (since it's first node)
        update_frames = [f for f in frames if isinstance(f, LLMMessagesUpdateFrame)]
        self.assertEqual(len(update_frames), 1, "Should have exactly one UpdateFrame")

        # Verify message content
        expected_messages = initial_node["role_messages"] + initial_node["task_messages"]
        actual_messages = update_frames[0].messages
        self.assertEqual(actual_messages, expected_messages)

    async def test_static_flow_transitions(self):
        """Test transitions in static flows.

        Verifies that:
        1. Static transitions correctly change the current node
        2. Node configuration is properly processed during transition
        3. Messages are sent using AppendFrame for non-initial nodes
        """
        # Setup flow manager with static configuration
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            tts=self.mock_tts,
            flow_config=self.static_flow_config,
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
        self.mock_task.queue_frames.assert_called_once()
        frames = self.mock_task.queue_frames.call_args[0][0]

        # Should have exactly one AppendFrame and one SetToolsFrame
        append_frames = [f for f in frames if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(len(append_frames), 1, "Should have exactly one AppendFrame")

    async def test_dynamic_flow_transitions(self):
        """Test transitions in dynamic flow.

        Verifies that:
        1. Transition callback is called with correct arguments
        2. Dynamic node transitions work properly
        3. State is updated correctly
        """
        # Create mock transition callback
        mock_transition_callback = AsyncMock()

        # Initialize flow manager with mock callback
        flow_manager = FlowManager(
            task=self.mock_task,
            llm=self.mock_llm,
            tts=self.mock_tts,
            transition_callback=mock_transition_callback,
        )
        await flow_manager.initialize()

        # Set initial node
        await flow_manager.set_node("start", self.sample_node)
        self.assertEqual(flow_manager.current_node, "start")

        # Reset frame tracking
        self.mock_task.queue_frames.reset_mock()

        # Trigger transition
        test_function_name = "test_function"
        test_args = {"test": "value"}
        await flow_manager.transition_callback(test_function_name, test_args, flow_manager)

        # Verify callback was called with correct arguments
        mock_transition_callback.assert_called_once_with(
            test_function_name, test_args, flow_manager
        )

    async def test_node_validation(self):
        """Test node configuration validation."""
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm, tts=self.mock_tts)
        await flow_manager.initialize()

        # Create node config with actions
        node_with_actions = {
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)

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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # Create node config with multiple functions
        node_config = {
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # Try to initialize again
        with patch("loguru.logger.warning") as mock_logger:
            await flow_manager.initialize()
            mock_logger.assert_called_once()

    async def test_register_action(self):
        """Test registering custom actions."""
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)

        async def custom_action(action):
            pass

        flow_manager.register_action("custom", custom_action)
        self.assertIn("custom", flow_manager.action_manager.action_handlers)

    async def test_call_handler_variations(self):
        """Test different handler signature variations."""
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
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
                "Function 'test_func' in node 'test' has neither handler nor transition_to",
                warning_message,
            )

    async def test_transition_callback_error_handling(self):
        """Test error handling in transition callback."""

        async def failing_transition(function_name, args, flow_manager):
            raise ValueError("Transition error")

        flow_manager = FlowManager(
            task=self.mock_task, llm=self.mock_llm, transition_callback=failing_transition
        )
        await flow_manager.initialize()

        transition_func = await flow_manager._create_transition_func(
            "test", None, transition_to=None
        )

        async def result_callback(result):
            pass

        await transition_func("test", "id", {}, None, None, result_callback)

    async def test_register_function_error_handling(self):
        """Test error handling in function registration."""
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # Mock LLM to raise error on register_function
        flow_manager.llm.register_function.side_effect = Exception("Registration error")

        new_functions = set()
        with self.assertRaises(FlowError):
            await flow_manager._register_function("test", None, None, new_functions)

    async def test_action_execution_error_handling(self):
        """Test error handling in action execution."""
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # Create node config with actions that will fail
        node_config = {
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # Mock task to raise error on queue_frames
        flow_manager.task.queue_frames.side_effect = Exception("Queue error")

        with self.assertRaises(FlowError):
            await flow_manager._update_llm_context(
                messages=[{"role": "system", "content": "Test"}], functions=[]
            )

    async def test_handler_callback_completion(self):
        """Test handler completion callback and logging."""
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        async def test_handler(args):
            return {"status": "success", "data": "test"}

        callback_called = False

        async def result_callback(result):
            nonlocal callback_called
            callback_called = True
            self.assertEqual(result["status"], "success")

        transition_func = await flow_manager._create_transition_func(
            "test", test_handler, transition_to=None
        )
        await transition_func("test", "id", {}, None, None, result_callback)
        self.assertTrue(callback_called)

    async def test_handler_removal_all_formats(self):
        """Test handler removal from different function configurations."""
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        async def test_handler(args):
            return {"status": "success"}

        # Create node config with OpenAI format for multiple functions
        node_config = {
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # Define test handler in main module
        async def test_handler_main(args):
            return {"status": "success"}

        # Add handler to main module
        import sys

        sys.modules["__main__"].test_handler_main = test_handler_main

        try:
            node_config = {
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        node_config = {
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
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # Define and register test handler
        test_called = False

        async def test_handler(args):
            nonlocal test_called
            test_called = True
            # Remove flow_manager from args before returning
            args_without_flow_manager = {k: v for k, v in args.items() if k != "flow_manager"}
            return {"status": "success", "args": args_without_flow_manager}

        import sys

        sys.modules["__main__"].test_handler = test_handler

        try:
            node_config = {
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

            async def callback(result):
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["args"], {"test": "value"})

            await func("test_function", "id", {"test": "value"}, None, None, callback)
            self.assertTrue(test_called)

        finally:
            delattr(sys.modules["__main__"], "test_handler")

    async def test_role_message_inheritance(self):
        """Test that role messages are properly handled between nodes.

        Verifies:
        1. Role messages are included in first node
        2. Role messages are included in subsequent nodes
        3. Messages are combined correctly
        """
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # First node with role messages
        first_node = {
            "role_messages": [{"role": "system", "content": "You are a helpful assistant."}],
            "task_messages": [{"role": "system", "content": "First task."}],
            "functions": [],
        }

        # Second node without role messages
        second_node = {
            "task_messages": [{"role": "system", "content": "Second task."}],
            "functions": [],
        }

        # Set first node and verify UpdateFrame
        await flow_manager.set_node("first", first_node)
        first_calls = self.mock_task.queue_frames.call_args_list[-1]
        first_frames = first_calls[0][0]
        update_frames = [f for f in first_frames if isinstance(f, LLMMessagesUpdateFrame)]
        self.assertEqual(len(update_frames), 1)

        # Verify combined messages in first node
        expected_first_messages = first_node["role_messages"] + first_node["task_messages"]
        self.assertEqual(update_frames[0].messages, expected_first_messages)

        # Reset mock and set second node
        self.mock_task.queue_frames.reset_mock()
        await flow_manager.set_node("second", second_node)

        # Verify AppendFrame for second node
        second_calls = self.mock_task.queue_frames.call_args_list[-1]
        second_frames = second_calls[0][0]
        append_frames = [f for f in second_frames if isinstance(f, LLMMessagesAppendFrame)]
        self.assertEqual(len(append_frames), 1)

        # Verify only task messages in second node
        self.assertEqual(append_frames[0].messages, second_node["task_messages"])

    async def test_frame_type_selection(self):
        """Test that correct frame types are used based on node order.

        Verifies:
        1. First node uses UpdateFrame
        2. Subsequent nodes use AppendFrame
        3. Frame content is correct
        """
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        test_node = {
            "task_messages": [{"role": "system", "content": "Test task."}],
            "functions": [],
        }

        # First node should use UpdateFrame
        await flow_manager.set_node("first", test_node)
        first_frames = self.mock_task.queue_frames.call_args[0][0]
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
        second_frames = self.mock_task.queue_frames.call_args[0][0]
        self.assertTrue(
            any(isinstance(f, LLMMessagesAppendFrame) for f in second_frames),
            "Subsequent nodes should use AppendFrame",
        )
        self.assertFalse(
            any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames),
            "Subsequent nodes should not use UpdateFrame",
        )

    async def test_state_references_in_content(self):
        """Test state references ($state.key) in message content."""
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # Set some state data
        flow_manager.state["orders"] = [{"type": "pizza", "price": 20.00}]

        # Create node with state reference
        node_config = {
            "task_messages": [{"role": "system", "content": "Order details: $state.orders"}],
            "functions": [],
        }

        await flow_manager.set_node("test", node_config)

        # Verify state was replaced in message
        calls = self.mock_task.queue_frames.call_args_list[-1]
        frames = calls[0][0]
        messages = next(f.messages for f in frames if hasattr(f, "messages"))
        expected_content = f"Order details: {flow_manager.state['orders']}"
        self.assertEqual(messages[0]["content"], expected_content)

    async def test_state_references_missing_key(self):
        """Test handling of missing state keys in content."""
        flow_manager = FlowManager(task=self.mock_task, llm=self.mock_llm)
        await flow_manager.initialize()

        # Create node referencing non-existent state
        node_config = {
            "task_messages": [{"role": "system", "content": "Data: $state.missing_key"}],
            "functions": [],
        }

        await flow_manager.set_node("test", node_config)

        # Verify missing key is handled gracefully
        calls = self.mock_task.queue_frames.call_args_list[-1]
        frames = calls[0][0]
        messages = next(f.messages for f in frames if hasattr(f, "messages"))
        self.assertEqual(messages[0]["content"], "Data: <no value for missing_key>")
