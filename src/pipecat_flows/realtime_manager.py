#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Realtime LLM flow management support.

This module provides RealtimeFlowManager, a FlowManager subclass that properly
handles node transitions for OpenAI/Azure Realtime LLM services.

The standard FlowManager queues LLMRunFrame to trigger responses, but Realtime
LLMs auto-respond when function results are sent - using the CURRENT session
instructions (old context). This causes stale context responses during transitions.

RealtimeFlowManager solves this by:
1. Pausing audio input during transitions
2. Pre-syncing new node instructions to the session
3. Waiting for session.updated confirmation
4. Triggering response with correct context

See: https://github.com/pipecat-ai/pipecat-flows/issues/66
"""

import asyncio
import time
from typing import Any, Dict, Optional

from loguru import logger

from pipecat_flows.manager import FlowManager
from pipecat_flows.types import ContextStrategy, ContextStrategyConfig, FlowConfig, NodeConfig

# Import Realtime LLM services for type checking
try:
    from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
    from pipecat.services.openai.realtime import events as realtime_events

    HAS_OPENAI_REALTIME = True
except ImportError:
    HAS_OPENAI_REALTIME, OpenAIRealtimeLLMService, realtime_events = False, None, None

try:
    from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService

    HAS_AZURE_REALTIME = True
except ImportError:
    HAS_AZURE_REALTIME, AzureRealtimeLLMService = False, None


class RealtimeFlowManager(FlowManager):
    """FlowManager with Realtime LLM transition handling.

    This subclass handles the special case of OpenAI/Azure Realtime LLMs which
    auto-respond to function results with the current session instructions. During
    node transitions, this causes the LLM to respond with stale context.

    Key behavior:
    - INITIAL node: Relies on Realtime API auto-response when user speaks.
    - TRANSITION nodes: Pauses audio, syncs new instructions, waits for
      session.updated confirmation, then triggers _create_response().

    Usage:
        from pipecat_flows import RealtimeFlowManager

        # Use for Realtime LLMs (OpenAI/Azure)
        flow_manager = RealtimeFlowManager(
            task=task,
            llm=realtime_llm,
            context_aggregator=context_aggregator
        )
        await flow_manager.initialize(initial_node)

    Attributes:
        SESSION_UPDATE_TIMEOUT: Maximum seconds to wait for session.updated event.
    """

    SESSION_UPDATE_TIMEOUT = 0.3  # seconds

    def __init__(self, *args, **kwargs):
        """Initialize the RealtimeFlowManager.

        Args:
            *args: Positional arguments passed to FlowManager.
            **kwargs: Keyword arguments passed to FlowManager.
        """
        super().__init__(*args, **kwargs)
        self._is_initial_setup = True
        self._session_update_future: Optional[asyncio.Future] = None

    async def initialize(self, initial_node: Optional[NodeConfig] = None) -> None:
        """Initialize the flow with the initial node.

        Marks that we're in initial setup so _set_node knows not to manually
        trigger a response (initial node relies on Realtime API auto-response).

        Args:
            initial_node: Optional initial node configuration for dynamic flows.
        """
        self._is_initial_setup = True
        try:
            await super().initialize(initial_node)
        finally:
            self._is_initial_setup = False

    def _is_realtime_llm(self) -> bool:
        """Check if the LLM is a Realtime service (OpenAI or Azure).

        Returns:
            True if the LLM is an OpenAI or Azure Realtime service.
        """
        return (HAS_OPENAI_REALTIME and isinstance(self._llm, OpenAIRealtimeLLMService)) or (
            HAS_AZURE_REALTIME and isinstance(self._llm, AzureRealtimeLLMService)
        )

    async def _wait_for_session_updated(self) -> bool:
        """Wait for session.updated event confirmation from Realtime API.

        This replaces a hardcoded sleep delay with an event-driven approach.
        The Realtime API sends a session.updated event after processing a
        session.update request. We wait for this confirmation before triggering
        the response, ensuring the new context is ready.

        Returns:
            True if session.updated was received, False if timeout occurred.
        """
        start = time.monotonic()
        self._session_update_future = asyncio.get_event_loop().create_future()
        original_handler = self._llm._handle_evt_session_updated

        async def handler(evt):
            await original_handler(evt)
            if self._session_update_future and not self._session_update_future.done():
                self._session_update_future.set_result(True)

        self._llm._handle_evt_session_updated = handler
        try:
            await asyncio.wait_for(self._session_update_future, timeout=self.SESSION_UPDATE_TIMEOUT)
            logger.debug(f"session.updated confirmed in {(time.monotonic() - start) * 1000:.0f}ms")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"session.updated timeout after {self.SESSION_UPDATE_TIMEOUT}s")
            return False
        finally:
            self._llm._handle_evt_session_updated = original_handler
            self._session_update_future = None

    async def _sync_instructions(self, node_config: NodeConfig) -> None:
        """Sync role_messages and task_messages to Realtime session instructions.

        The Realtime API uses a single 'instructions' field in the session config,
        not multiple system messages like the Chat API.

        Args:
            node_config: The node configuration containing role_messages and task_messages.
        """
        messages = [
            msg.get("content", "")
            for key in ("role_messages", "task_messages")
            for msg in node_config.get(key, [])
            if msg.get("content")
        ]
        if messages:
            self._llm._session_properties.instructions = "\n\n".join(messages)
            await self._llm._update_settings()

    async def _execute_transition(self, transition_info: Dict[str, Any]) -> None:
        """Execute the stored transition with pre-sync for Realtime LLMs.

        For Realtime LLMs:
        1. Pauses audio input to prevent auto-responses during transition
        2. Clears any pending audio buffer
        3. Syncs new node's instructions BEFORE parent processes transition

        Args:
            transition_info: Dictionary containing transition details.
        """
        if not self._is_realtime_llm():
            return await super()._execute_transition(transition_info)

        # Pause audio and clear buffer to prevent race conditions
        self._llm.set_audio_input_paused(True)
        if realtime_events:
            try:
                await self._llm.send_client_event(realtime_events.InputAudioBufferClearEvent())
            except Exception:
                pass

        # Get next node config and pre-sync instructions
        next_node = transition_info.get("next_node") or transition_info.get("transition_to")
        node_config = next_node if isinstance(next_node, dict) else self._nodes.get(next_node)
        if node_config:
            await self._sync_instructions(node_config)

        await super()._execute_transition(transition_info)

    async def _set_node(self, node_id: str, node_config: NodeConfig) -> None:
        """Override to handle Realtime LLM response triggering and context strategy.

        For Realtime LLMs:
        - Forces context_strategy=RESET to replace context (not append)
        - For TRANSITIONS: Waits for session.updated, resumes audio, triggers response
        - For INITIAL: Relies on Realtime API auto-response when user speaks

        Args:
            node_id: Identifier for the node.
            node_config: Configuration for the node.
        """
        if not self._is_realtime_llm() or not node_config.get("respond_immediately", True):
            return await super()._set_node(node_id, node_config)

        # Force RESET strategy and disable respond_immediately
        original_strategy = node_config.get("context_strategy")
        node_config["context_strategy"] = ContextStrategyConfig(strategy=ContextStrategy.RESET)
        node_config["respond_immediately"] = False

        try:
            await super()._set_node(node_id, node_config)
        finally:
            node_config["respond_immediately"] = True
            if original_strategy:
                node_config["context_strategy"] = original_strategy
            else:
                node_config.pop("context_strategy", None)

        # For transitions: wait for confirmation, resume audio, trigger response
        if not self._is_initial_setup:
            await self._wait_for_session_updated()
            self._llm.set_audio_input_paused(False)
            await self._llm._create_response()
