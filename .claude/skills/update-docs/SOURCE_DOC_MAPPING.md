# Source-to-Doc Mapping

Maps pipecat-flows source files to their documentation pages in the
[pipecat-ai/docs](https://github.com/pipecat-ai/docs) repository.

## Direct Mapping

| Source file | Doc page(s) | Notes |
|---|---|---|
| `types.py` | `server/frameworks/flows/types.mdx` | NodeConfig, FlowsFunctionSchema, ActionConfig, ContextStrategy, ContextStrategyConfig, type aliases, flows_direct_function decorator |
| `manager.py` | `server/frameworks/flows/flow-manager.mdx` | FlowManager constructor, properties, methods |
| `actions.py` | `server/frameworks/flows/flow-manager.mdx` (register_action), `server/frameworks/flows/types.mdx` (ActionConfig) | Built-in action types and custom action registration |
| `adapters.py` | `server/frameworks/flows/pipecat-flows.mdx` | LLM Provider Support table |
| `exceptions.py` | `server/frameworks/flows/exceptions.mdx` | Exception hierarchy and descriptions |

## Guide Impact

Changes to **any** source file may also affect:

| Guide page | What to check |
|---|---|
| `guides/features/pipecat-flows.mdx` | Code examples, NodeConfig property list, function definition examples, action examples, context strategy examples |

### Specific guide sections by source file

- **types.py** — Technical Overview (NodeConfig properties list), Functions section (FlowsFunctionSchema examples), Context Strategy section
- **manager.py** — Initialization section, Cross-Node Logic section (state, global_functions)
- **actions.py** — Actions section (built-in actions, custom actions, action timing)
- **adapters.py** — Cross-Provider Compatibility subsection under Messages
- **exceptions.py** — (rarely affects guide)

## Skip List

| Pattern | Reason |
|---|---|
| `__init__.py` | Re-exports only; no unique logic |
