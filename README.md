# Pipecat Flow Editor

A visual editor for creating and managing Pipecat conversation flows.

## Features

- Visual flow creation and editing
- Import/export of flow configurations
- Support for terminal and transitional functions
- Merge node support for complex flows
- Real-time validation

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)

### Installation

Clone the repository

```bash
git clone git@github.com:pipecat-ai/pipecat-flows.git
```

Navigate to project directory

```bash
cd pipecat-flows
```

Install dependencies

```bash
npm install
```

Start development server

```bash
npm run dev
```

Open the page in your browser at http://localhost:8080.

### Usage

1. Create a new flow using the toolbar buttons
2. Add nodes by right-clicking in the canvas
3. Connect nodes by dragging from outputs to inputs
4. Edit node properties in the side panel
5. Export your flow configuration using the toolbar

## Development

### Project Structure

```
pipecat-flows/
├── js/
│   ├── nodes/        # Node type definitions
│   ├── editor/       # Editor components
│   └── utils/        # Utility functions
├── css/             # Styles
└── examples/        # Example flows
```

### Available Scripts

- `npm start` - Start production server
- `npm run dev` - Start development server
- `npm run lint` - Check for linting issues
- `npm run lint:fix` - Fix linting issues
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting

### Adding New Node Types

TBD

### Contributing

TBD

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Reach us on X](https://x.com/pipecat_ai)
