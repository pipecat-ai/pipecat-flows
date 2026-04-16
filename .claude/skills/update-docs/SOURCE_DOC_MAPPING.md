# Source-to-Doc Mapping

Maps pipecat-flows source files to their documentation pages in the
[pipecat-ai/docs](https://github.com/pipecat-ai/docs) repository.

## API Reference Pages

| Source file | Doc page(s) | Notes |
|---|---|---|
| `types.py` | `api-reference/pipecat-flows/types.mdx` | NodeConfig, FlowsFunctionSchema, ActionConfig, ContextStrategy, ContextStrategyConfig, type aliases, flows_direct_function decorator |
| `manager.py` | `api-reference/pipecat-flows/flow-manager.mdx` | FlowManager constructor, properties, methods |
| `actions.py` | `api-reference/pipecat-flows/flow-manager.mdx` (register_action), `api-reference/pipecat-flows/types.mdx` (ActionConfig) | Built-in action types and custom action registration |
| `adapters.py` | `api-reference/pipecat-flows/overview.mdx` | LLM Provider Support table |
| `exceptions.py` | `api-reference/pipecat-flows/exceptions.mdx` | Exception hierarchy and descriptions |

## Guide Pages

Changes to source files may also affect the guide pages under `pipecat-flows/guides/`.

| Source file | Guide page(s) | What to check |
|---|---|---|
| `types.py` | `pipecat-flows/guides/nodes-and-messages.mdx` | NodeConfig properties, message format, respond_immediately |
| `types.py` | `pipecat-flows/guides/functions.mdx` | FlowsFunctionSchema examples, direct functions, flows_direct_function decorator |
| `types.py` | `pipecat-flows/guides/context-strategies.mdx` | ContextStrategy enum values, ContextStrategyConfig usage |
| `manager.py` | `pipecat-flows/guides/state-management.mdx` | FlowManager initialization, state dict, global_functions |
| `actions.py` | `pipecat-flows/guides/actions.mdx` | Built-in actions, custom actions, action timing |
| `adapters.py` | `pipecat-flows/guides/nodes-and-messages.mdx` | Cross-Provider Compatibility section |
| `exceptions.py` | (rarely affects guides) | |

## Other Pages

| Page | When to check |
|---|---|
| `pipecat-flows/introduction.mdx` | Rarely changes; only if installation or high-level framing changes |
| `pipecat-flows/guides/quickstart.mdx` | If FlowManager init signature, FlowsFunctionSchema, or handler return types change |
| `pipecat-flows/examples.mdx` | Only if examples are added or removed |
| `pipecat-flows/migration/migration-1.0.mdx` | Only on major API removals or new deprecations |

## Skip List

| Pattern | Reason |
|---|---|
| `__init__.py` | Re-exports only; no unique logic |
