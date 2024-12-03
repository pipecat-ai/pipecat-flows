# Changelog

All notable changes to **Pipecat Flows** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Extended the FlowManager to support both static and dynamic conversation
  flows
- Added new dynamic flow examples for OpenAI, Anthropic, and Gemini
- New type system with FlowArgs and FlowResult for better type safety

### Changed

- Simplified function handler registration process
- Updated all examples to use the new unified FlowManager

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
