# Streaming Markdown Parser

A TypeScript-based streaming Markdown parser that efficiently handles GitHub-flavored Markdown features and processes text in random-sized chunks while maintaining proper formatting.

## Features

- Real-time streaming markdown processing
- GitHub-style markdown formatting
- Handles complex edge cases:
  - Proper backtick handling (single, double, triple)
  - Ordered and unordered lists
  - Headers (h1, h2, h3)
  - Code blocks and inline code
  - Bold and italic text

## Installation

```bash
npm install
```

## Usage

The library is designed with modularity in mind, separating core rendering logic from streaming functionality. This allows you to use the parser with your own streaming implementation.

### Using with Built-in Streaming

If you want to use our complete solution:

```typescript
import { MarkdownStreamer, streamMarkdown } from './dist/markdown-streamer-simple.js';

// Create a new streamer instance
const container = document.getElementById('output');
const streamer = new MarkdownStreamer(container);

// Stream markdown content
streamMarkdown(
  markdownText,
  (chunk, isLast) => {
    streamer.processChunk(chunk);
    if (isLast) {
      streamer.finalize();
    }
  },
  () => {
    console.log('Streaming complete');
  }
);
```

### Using with Your Own Streaming Implementation

If you have your own streaming logic, you can use just the core renderer:

```typescript
import { MarkdownRenderer } from './dist/markdown-streamer-simple.js';

// Create a new renderer instance
const container = document.getElementById('output');
const renderer = new MarkdownRenderer(container);

// Your custom streaming code
yourStreamImplementation({
  onChunkReceived: (chunk) => {
    // Process each chunk through the renderer
    renderer.processChunk(chunk);
  },
  onComplete: () => {
    // Make sure to finalize when streaming is complete
    renderer.finalize();
  }
});
```

### Configuration Options

The renderer accepts configuration options:

```typescript
// Available options
const options = {
  preserveNumbering: true // Preserves original numbers in ordered lists
};

// Pass options to the renderer
const renderer = new MarkdownRenderer(container, options);
```

## Development

```bash
# Build TypeScript
npm run build

# Start development server
npx http-server
```

## Implementation Details

The parser uses a sophisticated state machine to handle streaming markdown content:

1. **Buffer Management**: Accumulates text chunks until complete lines are detected
2. **Line Processing**: Analyzes each complete line for markdown patterns
3. **State Tracking**: Maintains formatting state across chunks
4. **DOM Updates**: Efficiently updates the document as content arrives

### Architecture

The implementation is split into two main classes:

- **MarkdownRenderer**: Core rendering logic, processes markdown and updates the DOM
- **MarkdownStreamer**: Thin wrapper around the renderer that simplifies integration with streaming

This separation allows you to use the core rendering logic with any streaming implementation.

## License

MIT
