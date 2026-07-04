#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Pipecat Flows - Structured conversation framework for Pipecat.

This package provides a framework for building structured conversations in Pipecat.
The FlowManager handles conversation flows with support for state management,
function calling, and cross-provider compatibility.

Pipecat Flows determines conversation structure at runtime, supporting function
calling, action execution, and seamless transitions between conversation states.
"""

import warnings

from .exceptions import (
    ActionError,
    FlowError,
    FlowInitializationError,
    FlowTransitionError,
    InvalidFunctionError,
)
from .manager import FlowManager
from .types import (
    ActionConfig,
    ConsolidatedFunctionResult,
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowFunctionHandler,
    FlowResult,
    FlowsDirectFunction,
    FlowsFunctionSchema,
    LegacyFunctionHandler,
    NodeConfig,
    ZeroArgFunctionHandler,
    flows_direct_function,
    flows_tool_options,
)

warnings.warn(
    "`pipecat-ai-flows` is deprecated: Pipecat Flows is now part of `pipecat-ai`. "
    "Import from `pipecat.flows` instead (e.g. `from pipecat.flows import FlowManager`). "
    "This standalone package is frozen at its final release and will not receive further "
    "updates.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Flow Manager
    "FlowManager",
    # Types
    "ActionConfig",
    "ContextStrategy",
    "ContextStrategyConfig",
    "FlowArgs",
    "FlowFunctionHandler",
    "FlowResult",
    "ConsolidatedFunctionResult",
    "FlowsFunctionSchema",
    "LegacyFunctionHandler",
    "FlowsDirectFunction",
    "NodeConfig",
    "ZeroArgFunctionHandler",
    "flows_tool_options",
    "flows_direct_function",
    # Exceptions
    "FlowError",
    "FlowInitializationError",
    "FlowTransitionError",
    "InvalidFunctionError",
    "ActionError",
]
