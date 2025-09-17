# Changelog

All notable changes to **Pipecat Flows** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.21] - 2025-09-17

### Added

- Add support for the new Pipecat `LLMSwitcher`, which can be used as a drop-in
  replacement for `LLMService`s in scenarios where you want to switch LLMs at
  runtime.

  There are a couple of pre-requisites to using `LLMSwitcher`:

  - You must be using the new universal `LLMContext` and
    `LLMContextAggregatorPair` (as of Pipecat 0.0.82, supported only by
    Pipecat's OpenAI and Google LLM implementations, but with more on the way).
  - You must be using "direct" functions or `FlowsFunctionSchema` functions (as
    opposed to provider-specific formats).

  Using `LLMSwitcher` looks like this:

  ```python
  # Create shared context and aggregators for your LLM services
  context = LLMContext()
  context_aggregator = LLMContextAggregatorPair(context)

  # Instantiate your LLM services
  llm_openai = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
  llm_google = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

  # Instantiate a switcher
  # (ServiceSwitcherStrategyManual defaults to OpenAI, as it's first in the list)
  llm_switcher = LLMSwitcher(
      llms=[llm_openai, llm_google], strategy_type=ServiceSwitcherStrategyManual
  )

  # Create your pipeline as usual (passing the switcher instead of an LLM)
  pipeline = Pipeline(
    [
        transport.input(),
        stt,
        context_aggregator.user(),
        llm_switcher,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ]
  )
  task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

  # Initialize your flow manager as usual (passing the switcher instead of an LLM)
  flow_manager = FlowManager(
      task=task,
      llm=llm_switcher,
      context_aggregator=context_aggregator,
  )

  # ...
  # Start your flow as usual
  @transport.event_handler("on_client_connected")
  async def on_client_connected(transport, participant):
      await flow_manager.initialize(create_main_node())

  # ...
  # Whenever is appropriate, switch LLMs!
  await task.queue_frames([ManuallySwitchServiceFrame(service=llm_google)])
  ```

## [0.0.20] - 2025-08-27

### Changed

- Added an `@property` for the following `FlowManager` attributes in order to
  officially make them part of the public API: `state`, `task`, `transport`,
  and `current_node`.

### Fixed

- Corrected an issue where the Flow Manager state was marked as private. To
  make this clear, added a `@property` to access the state.

## [0.0.19] - 2025-08-25

### Deprecated

- Static Flows are now deprecated and will be removed in v1.0.0. Use Dynamic
  Flows in their place. The deprecation includes the `flow_config` arg of the
  `FlowManager` and the `FlowConfig` type.

## [0.0.18] - 2025-06-27

### Added

- Addded a new optional `name` field to `NodeConfig`. When using dynamic flows
  alongside "consolidated" functions that return a tuple (result, next node),
  giving the next node a `name` is helpful for debug logging. If you don't
  specify a `name`, an automatically-generated UUID is used.

- Added support for providing "consolidated" functions, which are responsible
  for both doing some work as well as specifying the next node to transition
  to. When using consolidated functions, you don't specify `transition_to` or
  `transition_callback`.

  Usage:

  ```python
  # "Consolidated" function
  async def do_something(args: FlowArgs) -> tuple[FlowResult, NodeConfig]:
    foo = args["foo"]
    bar = args.get("bar", "")

    # Do some work (optional; this function may be a transition-only function)
    result = await process(foo, bar)

    # Specify next node (optional; this function may be a work-only function)
    # This is either a NodeConfig (for dynamic flows) or a node name (for
    # static flows)
    next_node = create_another_node()

    return result, next_node

  def create_a_node() -> NodeConfig:
    return NodeConfig(
        task_messages=[
          # ...
        ],
        functions=[FlowsFunctionSchema(
            name="do_something",
            description="Do something interesting.",
            handler=do_something,
            properties={
              "foo": {
                "type": "integer",
                "description": "The foo to do something interesting with."
              },
              "bar": {
                "type": "string",
                "description": "The bar to do something interesting with."
              }
            },
            required=["foo"],
        )],
    )
  ```

- Added support for providing "direct" functions, which don't need an
  accompanying `FlowsFunctionSchema` or function definition dict. Instead,
  metadata (i.e. `name`, `description`, `properties`, and `required`) are
  automatically extracted from a combination of the function signature and
  docstring.

  Usage:

  ```python
  # "Direct" function
  # `flow_manager` must be the first parameter
  async def do_something(flow_manager: FlowManager, foo: int, bar: str = "") -> tuple[FlowResult, NodeConfig]:
    """
    Do something interesting.

    Args:
      foo (int): The foo to do something interesting with.
      bar (string): The bar to do something interesting with.
    """

    # Do some work (optional; this function may be a transition-only function)
    result = await process(foo, bar)

    # Specify next node (optional; this function may be a work-only function)
    # This is either a NodeConfig (for dynamic flows) or a node name (for static flows)
    next_node = create_another_node()

    return result, next_node

  def create_a_node() -> NodeConfig:
    return NodeConfig(
      task_messages=[
        # ...
      ],
      functions=[do_something]
    )
  ```

### Changed

- `functions` are now optional in the `NodeConfig`. Additionally, for AWS
  Bedrock, Anthropic, and Gemini, you no longer need to provide a no_op
  function. The LLM adapters now handle this case on your behalf. This allows
  you to either omit `functions` for nodes, which is common for the end node,
  or specify an empty function call list, if desired.

### Deprecated

- The `tts` parameter in `FlowManager.__init__()` is now deprecated and will
  be removed in a future version. The `tts_say` action now pushes a
  `TTSSpeakFrame`.

- Deprecated `transition_to` and `transition_callback` in favor of
  "consolidated" `handler`s that return a tuple (result, next node).
  Alternatively, you could use "direct" functions and avoid using
  `FlowsFunctionSchema`s or function definition dicts entirely. See the "Added"
  section above for more details.

- Deprecated `set_node()` in favor of doing the following for dynamic flows:

  - Prefer "consolidated" or "direct" functions that return a tuple (result,
    next node) over deprecated `transition_callback`s
  - Pass your initial node to `FlowManager.initialize()`
  - If you really need to set a node explicitly, use `set_node_from_config()`

  In all of these cases, you can provide a `name` in your new node's config for
  debug logging purposes.

### Fixed

- Fixed an issue where `RESET_WITH_SUMMARY` wasn't working for the
  `GeminiAdapter`. Now, the `GeminiAdapter` uses the `google-genai` package,
  aligning with the package used by `pipecat-ai`.

- Fixed an issue where if `run_in_parallel=False` was set for the LLM, the bot
  would trigger N completions for each sequential function call. Now, Flows
  uses Pipecat's internal function tracking to determine when there are more
  edge functions to call.

- Overhauled `pre_actions` and `post_actions` timing logic, making their timing
  more predictable and eliminating some bugs. For example, now `tts_say`
  actions will always run after the bot response, when used in `post_actions`.

## [0.0.17] - 2025-05-16

### Added

- Added support for `AWSBedrockLLMService` by adding an `AWSBedrockAdapter`.

### Changed

- Added `respond_immediately` to `NodeConfig`. Setting it to `False` has the
  effect of making the bot wait, after the node is activated, for the user to
  speak before responding. This can be used for the initial node, if you want
  the user to speak first.

- Bumped the minimum required `pipecat-ai` version to 0.0.67 to align with AWS
  Bedrock additions in Pipecat. This also adds support for `FunctionCallParams`
  which were added in 0.0.66.

- Updated to use `FunctionCallParams` as args for the function handler.

- Updated imports to use the new .stt, .llm, and .tts paths.

### Other

- Added AWS Bedrock examples for insurance and patient_intake.

- Updated examples to `audio_in_enabled=True` and remove `vad_enabled` and
  `vad_audio_passthrough` to align with the latest Pipecat `TransportParams`.

## [0.0.16] - 2025-03-26

### Added

- Added a new "function" action type, which queues a function to run "inline"
  in the pipeline (i.e. when the pipeline is done with all the work queued
  before it).

  This is useful for doing things at the end of the bot's turn.

  Example usage:

  ```python
  async def after_the_fun_fact(action: dict, flow_manager: FlowManager):
    print("Done telling the user a fun fact.")

  def create_node() -> NodeConfig:
    return NodeConfig(
      task_messages=[
        {
          "role": "system",
          "content": "Greet the user and tell them a fun fact."
        },
        post_actions=[
          ActionConfig(
            type="function",
            handler=after_the_fun_fact
          )
        ]
      ]
    )
  ```

- Added support for `OpenAILLMService` subclasses in the adapter system. You
  can now use any Pipecat LLM service that inherits from `OpenAILLMService`
  such as `AzureLLMService`, `GrokLLMService`, `GroqLLMService`, and other
  without requiring adapter updates. See the Pipecat docs for
  [supported LLM services](https://docs.pipecat.ai/server/services/supported-services#large-language-models).

- Added a new `FlowsFunctionSchema` class, which allows you to specify function
  calls using a standard schema. This is effectively a subclass of Pipecat's
  `FunctionSchema`.

Example usage:

```python
# Define a function using FlowsFunctionSchema
collect_name = FlowsFunctionSchema(
    name="collect_name",
    description="Record the user's name",
    properties={
        "name": {"type": "string", "description": "The user's name"}
    },
    required=["name"],
    handler=collect_name_handler,
    transition_to="next_node"
)

# Use in node configuration
node_config = {
    "task_messages": [...],
    "functions": [collect_name]
}
```

### Changed

- Function handlers can now receive either `FlowArgs` only (legacy style) or
  both `FlowArgs` and the `FlowManager` instance (modern style). Adding support
  for the `FlowManager` provides access to conversation state, transport
  methods, and other flow resources within function handlers. The framework
  automatically detects which signature you're using and calls handlers
  appropriately.

### Dependencies

- Updated minimum Pipecat version to 0.0.60 to use `FunctionSchema` and
  provider-specific adapters.

### Other

- Update restaurant_reservation.py and insurance_gemini.py to use
  `FlowsFunctionSchema`.

- Updated examples to specify a `params` arg for `PipelineTask`, meeting the
  Pipecat requirement starting 0.0.58.

## [0.0.15] - 2025-02-26

### Changed

- The `ActionManager` now has access to the `FlowManager`, allowing more
  flexibility to create custom actions.

### Fixed

- Fixed an issue with the LLM adapter where you were required to install all
  LLM dependencies to run Flows.

## [0.0.14] - 2025-02-08

### Reverted

- Temporarily reverted the deprecation of the `tts` parameter in
  `FlowManager.__init__()`. This feature will be deprecated in a future release
  after the required Pipecat changes are completed.

## [0.0.13] - 2025-02-06

### Added

- Added context update strategies to control how context is managed during node
  transitions:
  - `APPEND`: Add new messages to existing context (default behavior)
  - `RESET`: Clear and replace context with new messages and most recent
    function call results
  - `RESET_WITH_SUMMARY`: Reset context but include an LLM-generated summary
    along with the new messages
  - Strategies can be set globally or per-node
  - Includes automatic fallback to RESET if summary generation fails

Example usage:

```python
# Global strategy
flow_manager = FlowManager(
    context_strategy=ContextStrategyConfig(
        strategy=ContextStrategy.RESET
    )
)

# Per-node strategy
node_config = {
    "task_messages": [...],
    "functions": [...],
    "context_strategy": ContextStrategyConfig(
        strategy=ContextStrategy.RESET_WITH_SUMMARY,
        summary_prompt="Summarize the key points discussed so far."
    )
}
```

- Added a new function called `get_current_context` which provides access to
  the LLM context.

Example usage:

```python
# Access current conversation context
context = flow_manager.get_current_context()
```

- Added a new dynamic example called `restaurant_reservation.py`.

### Changed

- Transition callbacks now receive function results directly as a second argument:
  `async def handle_transition(args: Dict, result: FlowResult, flow_manager: FlowManager)`.
  This enables direct access to typed function results for making routing decisions.
  For backwards compatibility, the two-argument signature
  `(args: Dict, flow_manager: FlowManager)` is still supported.

- Updated dynamic examples to use the new result argument.

### Deprecated

- The `tts` parameter in `FlowManager.__init__()` is now deprecated and will
  be removed in a future version. The `tts_say` action now pushes a
  `TTSSpeakFrame`.

## [0.0.12] - 2025-01-30

### Added

- Support for inline action handlers in flow configuration:
  - Actions can now be registered via handler field in config
  - Maintains backwards compatibility with manual registration
  - Built-in actions (`tts_say`, `end_conversation`) work without changes

Example of the new pattern:

```python
"pre_actions": [
    {
        "type": "check_status",
        "handler": check_status_handler
    }
]
```

### Changed

- Updated dynamic flows to use per-function, inline transition callbacks:
  - Removed global `transition_callback` from FlowManager initialization
  - Transition handlers are now specified directly in function definitions
  - Dynamic transitions are now specified similarly to the static flows'
    `transition_to` field
  - **Breaking change**: Dynamic flows must now specify transition callbacks in
    function configuration

Example of the new pattern:

```python
# Before - global transition callback
flow_manager = FlowManager(
    transition_callback=handle_transition
)

# After - inline transition callbacks
def create_node() -> NodeConfig:
    return {
        "functions": [{
            "type": "function",
            "function": {
                "name": "collect_age",
                "handler": collect_age,
                "description": "Record user's age",
                "parameters": {...},
                "transition_callback": handle_age_collection
            }
        }]
    }
```

- Updated dynamic flow examples to use the new `transition_callback` pattern.

### Fixed

- Fixed an issue where multiple, consecutive function calls could result in two completions.

## [0.0.11] - 2025-01-19

### Changed

- Updated `FlowManager` to more predictably handle function calls:

  - Edge functions (which transition to a new node) now result in an LLM
    completion after both the function call and messages are added to the
    LLM's context.
  - Node functions (which execute a function call without transitioning nodes)
    result in an LLM completion upon the function call result returning.
  - This change also improves the reliability of the pre- and post-action
    execution timing.

- Breaking changes:

  - The FlowManager has a new required arg, `context_aggregator`.
  - Pipecat's minimum version has been updated to 0.0.53 in order to use the
    new `FunctionCallResultProperties` frame.

- Updated all examples to align with the new changes.

## [0.0.10] - 2024-12-21

### Changed

- Nodes now have two message types to better delineate defining the role or
  persona of the bot from the task it needs to accomplish. The message types are:

  - `role_messages`, which defines the personality or role of the bot
  - `task_messages`, which defines the task to be completed for a given node

- `role_messages` can be defined for the initial node and then inherited by
  subsequent nodes. You can treat this as an LLM "system" message.

- Simplified FlowManager initialization by removing the need for manual context
  setup in both static and dynamic flows. Now, you need to create a `FlowManager`
  and initialize it to start the flow.
- All examples have been updated to align with the API changes.

### Fixed

- Fixed an issue where importing the Flows module would require OpenAI,
  Anthropic, and Google LLM modules.

## [0.0.9] - 2024-12-08

### Changed

- Fixed function handler registration in FlowManager to handle `__function__:` tokens
  - Previously, the handler string was used directly, causing "not callable" errors
  - Now correctly looks up and uses the actual function object from the main module
  - Supports both direct function references and function names exported from the Flows editor

## [0.0.8] - 2024-12-07

### Changed

- Improved type safety in FlowManager by requiring keyword arguments for initialization
- Enhanced error messages for LLM service type validation

## [0.0.7] - 2024-12-06

### Added

- New `transition_to` field for static flows
  - Combines function handlers with state transitions
  - Supports all LLM providers (OpenAI, Anthropic, Gemini)
  - Static examples updated to use this new transition

### Changed

- Static flow transitions now use `transition_to` instead of matching function names
  - Before: Function name had to match target node name
  - After: Function explicitly declares target via `transition_to`

### Fixed

- Duplicate LLM responses during transitions

## [0.0.6] - 2024-12-02

### Added

- New FlowManager supporting both static and dynamic conversation flows
  - Static flows: Configuration-driven with predefined paths
  - Dynamic flows: Runtime-determined conversation paths
  - Documentation: [Guide](https://docs.pipecat.ai/guides/pipecat-flow) and [Reference](https://docs.pipecat.ai/api-reference/utilities/flows/pipecat-flows)
- Provider-specific examples demonstrating dynamic flows:
  - OpenAI: `insurance_openai.py`
  - Anthropic: `insurance_anthropic.py`
  - Gemini: `insurance_gemini.py`
- Type safety improvements:
  - `FlowArgs`: Type-safe function arguments
  - `FlowResult`: Type-safe function returns

### Changed

- Simplified function handling:
  - Automatic LLM function registration
  - Optional handlers for edge nodes
- Updated all examples to use unified FlowManager interface

## [0.0.5] - 2024-11-27

### Added

- Added LLM support for:

  - Anthropic
  - Google Gemini

- Added `LLMFormatParser`, a format parser to handle LLM provider-specific
  messages and function call formats

- Added new examples:

  - movie_explorer_anthropic.py (Claude 3.5)
  - movie_explorer_gemini.py (Gemini 1.5 Flash)
  - travel_planner_gemini.py (Gemini 1.5 Flash)

## [0.0.4] - 2024-11-26

### Added

- New example `movie_explorer.py` demonstrating:
  - Real API integration with TMDB
  - Node functions for API calls
  - Edge functions for state transitions
  - Proper function registration pattern

### Changed

- Renamed function types to use graph terminology:

  - "Terminal functions" are now "node functions" (operations within a state)
  - "Transitional functions" are now "edge functions" (transitions between states)

- Updated function registration process:

  - Node functions must be registered directly with the LLM before flow initialization
  - Edge functions are automatically registered by FlowManager during initialization
  - LLM instance is now required in FlowManager constructor

- Added flexibility to node naming with the Editor:
  - Start nodes can now use any descriptive name (e.g., "greeting")
  - End nodes conventionally use "end" but support custom names
  - Flow configuration's `initial_node` property determines the starting state

### Updated

- All examples updated to use new function registration pattern
- Documentation updated to reflect new terminology and patterns
- Editor updated to support flexible node naming

## [0.0.3] - 2024-11-25

### Added

- Added an `examples` directory which contains five different examples
  showing how to build a conversation flow with Pipecat Flows.

- Added a new editor example called `patient_intake.json` which demonstrates
  a patient intake conversation flow.

### Changed

- `pipecat-ai-flows` now includes `pipecat-ai` as a dependency, making it
  easier to get started with a fresh installation.

### Fixed

- Fixed an issue where terminal functions were updating the LLM context and
  tools. Now, only transitional functions update the LLM context and tools.

## [0.0.2] - 2024-11-22

### Fixed

- Fixed an issue where `pipecat-ai` was mistakenly added as a dependency

## [0.0.1] - 2024-11-18

### Added

- Initial public beta release.

- Added conversation flow management system through `FlowState` and `FlowManager` classes.
  This system enables developers to create structured, multi-turn conversations using
  a node-based state machine. Each node can contain:
  - Multiple LLM context messages (system/user/assistant)
  - Available functions for that state
  - Pre- and post-actions for state transitions
  - Support for both terminal functions (stay in same node) and transitional functions
  - Built-in handlers for immediate TTS feedback and conversation end
- Added `NodeConfig` dataclass for defining conversation states, supporting:
  - Multiple messages per node for complex prompt building
  - Function definitions for available actions
  - Optional pre- and post-action hooks
  - Clear separation between node configuration and state management
