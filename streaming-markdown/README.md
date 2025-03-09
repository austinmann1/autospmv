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

## License

MIT
