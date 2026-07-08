#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""JSON-based flow configuration adapter for Pipecat Flows.

Bridges a static JSON flow definition and Pipecat Flows. The flow graph
(nodes, messages, function schemas, transitions) lives in JSON, while
custom computation still happens in Python handler functions resolved from
a namespace dict (typically ``globals()``).

Function resolution rules:
    - transition_to only: auto-generates a handler that returns (None, next_node)
    - handler + transition_to: resolves handler from namespace, wraps result with next_node
    - handler only: resolves handler from namespace, no transition (returns result, None)
    - neither: raises ValueError at validation time
"""

import json
from pathlib import Path
from typing import Callable

from pipecat_flows import (
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)


class JSONFlowConfig:
    """Bridges a static JSON flow definition and Pipecat Flows.

    Reads a JSON config that defines the flow graph (nodes, messages, function
    schemas, transitions) and produces NodeConfig objects at runtime. Simple
    transitions are auto-generated; functions that need custom logic reference
    Python handlers by name, resolved from the provided namespace dict.

    Args:
        config: A dict, JSON string, or path to a JSON file.
        namespace: A dict mapping names to callables (typically ``globals()``).
            Handler strings in the JSON config are resolved against this dict
            lazily at node creation time.
    """

    def __init__(self, config: dict | str | Path, namespace: dict):
        self._config = self._load_config(config)
        self._namespace = namespace
        self._validate()

    def _load_config(self, config: dict | str | Path) -> dict:
        """Parse dict, JSON string, or file path into a config dict."""
        if isinstance(config, dict):
            return config
        if isinstance(config, Path):
            return json.loads(config.read_text())
        path = Path(config)
        if path.exists():
            return json.loads(path.read_text())
        return json.loads(config)

    def _validate(self) -> None:
        """Validate structural correctness at load time."""
        initial = self._config.get("initial_node")
        nodes = self._config.get("nodes", {})

        if not initial:
            raise ValueError("Config missing 'initial_node'")
        if initial not in nodes:
            raise ValueError(f"initial_node '{initial}' not found in nodes")

        # Collect all function definitions for validation
        all_functions: list[tuple[dict, str | None]] = []
        for node_name, node in nodes.items():
            for func in node.get("functions", []):
                all_functions.append((func, node_name))
        for func in self._config.get("global_functions", []):
            all_functions.append((func, None))

        for func, source in all_functions:
            has_handler = "handler" in func
            has_transition = "transition_to" in func
            location = f"node '{source}'" if source else "global functions"
            if not has_handler and not has_transition:
                raise ValueError(
                    f"Function '{func['name']}' in {location} "
                    "must have at least one of 'handler' or 'transition_to'"
                )
            if has_transition and func["transition_to"] not in nodes:
                raise ValueError(
                    f"Function '{func['name']}' in {location} "
                    f"references unknown node '{func['transition_to']}'"
                )

    def _build_function(self, func_def: dict) -> FlowsFunctionSchema:
        """Convert a JSON function definition into a FlowsFunctionSchema."""
        has_handler = "handler" in func_def
        has_transition = "transition_to" in func_def

        if has_handler and has_transition:
            handler = self._make_wrapped_handler(func_def["handler"], func_def["transition_to"])
        elif has_transition:
            handler = self._make_transition_handler(func_def["transition_to"])
        else:
            handler = self._make_passthrough_handler(func_def["handler"])

        return FlowsFunctionSchema(
            name=func_def["name"],
            handler=handler,
            description=func_def.get("description", ""),
            properties=func_def.get("properties", {}),
            required=func_def.get("required", []),
        )

    def _make_transition_handler(self, target_node: str) -> Callable:
        """Auto-generate a handler that just transitions to the target node."""

        async def handler(args: FlowArgs, flow_manager: FlowManager):
            return None, self.create_node(target_node)

        return handler

    def _make_wrapped_handler(self, handler_name: str, target_node: str) -> Callable:
        """Resolve a handler from the namespace and pair its result with the target node."""

        async def handler(args: FlowArgs, flow_manager: FlowManager):
            resolved = self._namespace.get(handler_name)
            if not resolved:
                raise ValueError(f"Handler '{handler_name}' not found in namespace")
            result = await resolved(args, flow_manager)
            return result, self.create_node(target_node)

        return handler

    def _make_passthrough_handler(self, handler_name: str) -> Callable:
        """Resolve a handler from the namespace that stays on the current node."""

        async def handler(args: FlowArgs, flow_manager: FlowManager):
            resolved = self._namespace.get(handler_name)
            if not resolved:
                raise ValueError(f"Handler '{handler_name}' not found in namespace")
            result = await resolved(args, flow_manager)
            return result, None

        return handler

    def _resolve_actions(self, actions: list[dict] | None) -> list[dict] | None:
        """Resolve string handler references in function-type actions to callables."""
        if not actions:
            return None
        resolved = []
        for action in actions:
            action = dict(action)  # Copy to avoid mutating the config
            if action.get("type") == "function" and isinstance(action.get("handler"), str):
                handler_name = action["handler"]
                handler = self._namespace.get(handler_name)
                if not handler:
                    raise ValueError(f"Action handler '{handler_name}' not found in namespace")
                action["handler"] = handler
            resolved.append(action)
        return resolved

    def create_node(self, node_name: str) -> NodeConfig:
        """Build a NodeConfig from the JSON definition for the given node."""
        node_def = self._config["nodes"][node_name]

        node: NodeConfig = {
            "name": node_name,
            "task_messages": node_def["task_messages"],
        }

        if "role_message" in node_def:
            node["role_message"] = node_def["role_message"]

        if "functions" in node_def:
            node["functions"] = [self._build_function(f) for f in node_def["functions"]]

        pre_actions = self._resolve_actions(node_def.get("pre_actions"))
        if pre_actions:
            node["pre_actions"] = pre_actions

        post_actions = self._resolve_actions(node_def.get("post_actions"))
        if post_actions:
            node["post_actions"] = post_actions

        if "context_strategy" in node_def:
            cs = node_def["context_strategy"]
            node["context_strategy"] = ContextStrategyConfig(
                strategy=ContextStrategy(cs["strategy"]),
                summary_prompt=cs.get("summary_prompt"),
            )

        if "respond_immediately" in node_def:
            node["respond_immediately"] = node_def["respond_immediately"]

        return node

    def create_initial_node(self) -> NodeConfig:
        """Build the NodeConfig for the initial node defined in the config."""
        return self.create_node(self._config["initial_node"])

    def create_global_functions(self) -> list[FlowsFunctionSchema]:
        """Build FlowsFunctionSchema objects for all global functions in the config."""
        return [self._build_function(f) for f in self._config.get("global_functions", [])]
