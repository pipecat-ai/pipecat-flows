#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Pipecat Flows - Structured conversation framework for Pipecat.

This package provides a framework for building structured conversations in Pipecat.
The FlowManager can handle both static and dynamic conversation flows with support
for state management, function calling, and cross-provider compatibility.

Static flows use predefined conversation paths configured upfront, while dynamic
flows determine conversation structure at runtime. Both approaches support function
calling, action execution, and seamless transitions between conversation states.
"""

from .exceptions import (
    ActionError,
    FlowError,
    FlowInitializationError,
    FlowTransitionError,
    InvalidFunctionError,
)
from .manager import FlowManager
from .types import (
    ConsolidatedFunctionResult,
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowConfig,
    FlowFunctionHandler,
    FlowResult,
    FlowsDirectFunction,
    FlowsFunctionSchema,
    LegacyFunctionHandler,
    NodeConfig,
)

__all__ = [
    # Flow Manager
    "FlowManager",
    # Types
    "ContextStrategy",
    "ContextStrategyConfig",
    "FlowArgs",
    "FlowConfig",
    "FlowFunctionHandler",
    "FlowResult",
    "ConsolidatedFunctionResult",
    "FlowsFunctionSchema",
    "LegacyFunctionHandler",
    "FlowsDirectFunction",
    "NodeConfig",
    # Exceptions
    "FlowError",
    "FlowInitializationError",
    "FlowTransitionError",
    "InvalidFunctionError",
    "ActionError",
]
