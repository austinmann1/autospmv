/**
 * Markdown Streamer - A TypeScript implementation of a streaming markdown parser
 * that handles GitHub-style markdown formatting with random-sized chunks
 */

/**
 * Sample markdown text for testing the streamer
 */
export const sampleMarkdown = `# This is an example of a text formatted in Markdown to be rendered

In this assignment, you'll be formatting this text as it comes from a \` stream, but you may need to figure out if it actually needs to be formatted.

1. Example one

In the following function, \`x = 4\`, which we see below:

\`\`\`python
def get_number():
    return 4
\`\`\`

2. Example two

We also can have \`\`\` a situation where we are given \`y = "some string"\`, like so:

\`\`\`javascript
function getString() {
    let y = "some string";
    return \`This is a string with a backtick: \${y}\`;
}
\`\`\`

## Edge cases

You also want to handle edge cases \`\`\` like this, where there are three backticks without closing backticks, and then maybe there's one backtick \`. There's no closing backtick on this line, so we don't want to render it as code.

## Further consideration

If time permits, try to implement some styling for other elements in markdown, like the header above, or these bullet points:

* That might have **bold text**
* However, this is **not** a core part of the assignment

## Final thoughts

* Make sure to be able to handle these \` edge cases
* Where there are many backticks for \`code\` but some that \`\`\` don't need to be styled \`
* Happy Coding :)`;

/**
 * Interface for the markdown rendering options
 */
export interface MarkdownRendererOptions {
  // Add any configuration options here
  preserveNumbering?: boolean;
}

/**
 * Core markdown renderer class - handles the actual markdown parsing logic
 * This class is independent of streaming/chunking and focuses solely on rendering
 */
export class MarkdownRenderer {
  private container: HTMLElement;
  private buffer: string;
  private inCode: boolean;
  private codeBlockCount: number;
  private inParagraph: boolean;
  private inListItem: boolean;
  private inHeading: boolean;
  private headingLevel: number;
  private options: MarkdownRendererOptions;

  /**
   * Create a new markdown renderer
   * @param container The HTML element to render markdown into
   * @param options Configuration options for the renderer
   */
  constructor(container: HTMLElement, options: MarkdownRendererOptions = {}) {
    this.container = container;
    this.options = options;
    this.buffer = '';
    this.inCode = false;
    this.codeBlockCount = 0;
    this.inParagraph = false;
    this.inListItem = false;
    this.inHeading = false;
    this.headingLevel = 0;
    this.reset();
  }

  /**
   * Reset the renderer state and clear the container
   */
  public reset(): void {
    this.container.innerHTML = '';
    this.buffer = '';
    this.inCode = false;
    this.codeBlockCount = 0;
    this.inParagraph = false;
    this.inListItem = false;
    this.inHeading = false;
    this.headingLevel = 0;
  }

  /**
   * Process a chunk of markdown text
   * @param chunk The chunk of text to process
   */
  public processChunk(chunk: string): void {
    // Add the chunk to our buffer
    this.buffer += chunk;
    
    // Process the buffer
    this.processBuffer();
  }

  /**
   * Process the current buffer content
   */
  private processBuffer(): void {
    // Check for complete lines in the buffer
    const lines = this.buffer.split('\n');
    
    // If we have more than one line, we can process all but the last one
    if (lines.length > 1) {
      // Process all complete lines
      for (let i = 0; i < lines.length - 1; i++) {
        this.processLine(lines[i]);
      }
      
      // Keep the last (potentially incomplete) line in the buffer
      this.buffer = lines[lines.length - 1];
    }
  }

  /**
   * Process a line of text to detect triple backticks not at the start of a line
   * @param line The line to process
   */
  private processLine(line: string): void {
    // Check for code blocks with triple backticks at the start of a line
    if (line.trim().startsWith('```')) {
      this.handleCodeBlock(line);
      return;
    }
    
    // If we're in a code block, just add the line as-is
    if (this.inCode) {
      const pre = this.container.querySelector('pre:last-child');
      if (pre) {
        const code = pre.querySelector('code');
        if (code) {
          code.textContent = (code.textContent || '') + line + '\n';
        }
      }
      return;
    }
    
    // Check for headings
    if (line.startsWith('# ')) {
      this.createHeading(line.substring(2), 1);
      return;
    } else if (line.startsWith('## ')) {
      this.createHeading(line.substring(3), 2);
      return;
    } else if (line.startsWith('### ')) {
      this.createHeading(line.substring(4), 3);
      return;
    }
    
    // Check for list items
    if (line.match(/^\s*[*-]\s+/)) {
      this.createListItem(line.replace(/^\s*[*-]\s+/, ''));
      return;
    }
    
    // Check for numbered list items
    const orderedListMatch = line.match(/^\s*(\d+)\.\s+(.*)/);
    if (orderedListMatch) {
      const number = parseInt(orderedListMatch[1]);
      const content = orderedListMatch[2];
      this.createNumberedListItem(content, number);
      return;
    }
    
    // If the line is empty, end any current paragraph
    if (line.trim() === '') {
      this.closeCurrentBlock();
      return;
    }
    
    // Otherwise, it's a normal paragraph
    this.addToParagraph(line);
  }
  
  /**
   * Handle code block delimiters (triple backticks)
   * @param line The line containing the code block delimiter
   */
  private handleCodeBlock(line: string): void {
    if (this.inCode) {
      // Closing a code block
      this.inCode = false;
    } else {
      // Opening a code block
      this.inCode = true;
      this.closeCurrentBlock();
      
      const language = line.trim().substring(3).trim();
      const pre = document.createElement('pre');
      const code = document.createElement('code');
      if (language) {
        code.className = `language-${language}`;
      }
      pre.appendChild(code);
      this.container.appendChild(pre);
    }
  }
  
  /**
   * Create a heading element
   * @param text The heading text
   * @param level The heading level (1-6)
   */
  private createHeading(text: string, level: number): void {
    this.closeCurrentBlock();
    
    const heading = document.createElement(`h${level}`);
    heading.innerHTML = this.formatInlineMarkdown(text);
    this.container.appendChild(heading);
  }
  
  /**
   * Create a bullet list item
   * @param text The list item text
   */
  private createListItem(text: string): void {
    const ul = this.getOrCreateList('ul');
    const li = document.createElement('li');
    li.innerHTML = this.formatInlineMarkdown(text);
    ul.appendChild(li);
  }
  
  /**
   * Create a numbered list item
   * @param text Text content for the list item
   * @param number The number to use for this list item
   */
  private createNumberedListItem(text: string, number: number = 1): void {
    let ol = this.container.querySelector('ol:last-child');
    
    // Check if we have a gap in content that should break the list
    const lastChild = this.container.lastElementChild;
    if (ol && lastChild !== ol) {
      // Create a new ordered list if there's content between this and the last list
      ol = null;
    }
    
    if (!ol) {
      // Create a new ordered list
      ol = document.createElement('ol');
      // Set the start attribute to preserve numbering
      ol.setAttribute('start', number.toString());
      this.container.appendChild(ol);
    } else if (number > 1) {
      // If this is not the first item and the number is greater than 1,
      // update the start attribute and adjust for existing items
      const existingItems = ol.querySelectorAll('li').length;
      if (existingItems === 1 && number === 2) {
        // This is the expected sequence, no need to adjust
      } else {
        // This is an unexpected sequence, adjust the start attribute
        ol.setAttribute('start', number.toString());
      }
    }
    
    const li = document.createElement('li');
    li.innerHTML = this.formatInlineMarkdown(text);
    ol.appendChild(li);
  }
  
  /**
   * Get or create a list element
   * @param listType The type of list ('ul' or 'ol')
   * @returns The list element
   */
  private getOrCreateList(listType: 'ul' | 'ol'): HTMLElement {
    // Check if the last element is already a list of the desired type
    const lastChild = this.container.lastElementChild;
    if (lastChild && lastChild.tagName.toLowerCase() === listType) {
      return lastChild as HTMLElement;
    }
    
    // Otherwise, create a new list
    const list = document.createElement(listType);
    this.container.appendChild(list);
    return list;
  }
  
  /**
   * Add text to a paragraph
   * @param text The text to add
   */
  private addToParagraph(text: string): void {
    if (!this.inParagraph) {
      // Start a new paragraph
      const p = document.createElement('p');
      this.container.appendChild(p);
      this.inParagraph = true;
    }
    
    const p = this.container.querySelector('p:last-child');
    if (p) {
      const content = p.innerHTML ? p.innerHTML + ' ' + this.formatInlineMarkdown(text) : this.formatInlineMarkdown(text);
      p.innerHTML = content;
    }
  }
  
  /**
   * Close any current block element
   */
  private closeCurrentBlock(): void {
    this.inParagraph = false;
  }
  
  /**
   * Format inline markdown elements
   * @param text The text to format
   * @returns Formatted HTML
   */
  private formatInlineMarkdown(text: string): string {
    // For inline code, we need a more advanced approach to handle complex backtick patterns
    // Process one character at a time to handle backticks properly
    let result = '';
    let i = 0;
    
    while (i < text.length) {
      // Check for inline code with single backticks
      if (text[i] === '`' && 
          !(i > 0 && text[i-1] === '`') && // Not part of double/triple backtick
          !(i < text.length - 1 && text[i+1] === '`')) { // Not part of double/triple backtick
        
        // Find the matching closing backtick
        let j = i + 1;
        while (j < text.length && text[j] !== '`') {
          j++;
        }
        
        if (j < text.length) { // We found a closing backtick
          // Extract the code content
          const codeContent = text.substring(i + 1, j);
          result += `<code>${codeContent}</code>`;
          i = j + 1; // Move past the closing backtick
          continue;
        }
      }
      
      // Check for triple backticks (only if not already handled as inline code)
      if (i + 2 < text.length && text[i] === '`' && text[i+1] === '`' && text[i+2] === '`') {
        // Treat as literal backticks in inline text (not a code block)
        result += '```';
        i += 3;
        continue;
      }
      
      // Check for double backticks
      if (i + 1 < text.length && text[i] === '`' && text[i+1] === '`') {
        // Treat as literal backticks
        result += '``';
        i += 2;
        continue;
      }
      
      // Regular character
      result += text[i];
      i++;
    }
    
    // Handle bold with double asterisks
    result = result.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Handle italic with single asterisk
    result = result.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    return result;
  }

  /**
   * Finalize the markdown processing
   */
  public finalize(): void {
    // Process any remaining content in the buffer
    if (this.buffer.trim()) {
      this.processLine(this.buffer);
    }
    
    // Close any open blocks
    this.closeCurrentBlock();
  }
}

/**
 * A class that uses MarkdownRenderer with streaming capabilities
 * This is a wrapper that combines the renderer with streaming logic
 */
export class MarkdownStreamer {
  private renderer: MarkdownRenderer;
  
  /**
   * Create a new markdown streamer
   * @param container The HTML element to render markdown into
   * @param options Configuration options for the renderer
   */
  constructor(container: HTMLElement, options: MarkdownRendererOptions = {}) {
    this.renderer = new MarkdownRenderer(container, options);
  }
  
  /**
   * Reset the streamer state and clear the container
   */
  public reset(): void {
    this.renderer.reset();
  }
  
  /**
   * Process a chunk of markdown text
   * @param chunk The chunk of text to process
   */
  public processChunk(chunk: string): void {
    this.renderer.processChunk(chunk);
  }
  
  /**
   * Finalize the markdown processing
   */
  public finalize(): void {
    this.renderer.finalize();
  }
}

/**
 * Simulate streaming of markdown text in chunks
 * @param text The markdown text to stream
 * @param onChunk Callback function for each chunk
 * @param onComplete Callback function when streaming is complete
 */
export function streamMarkdown(
  text: string,
  onChunk: (chunk: string, isLast: boolean) => void,
  onComplete: () => void
): void {
  let position = 0;
  const totalLength = text.length;
  
  const streamNextChunk = () => {
    if (position < totalLength) {
      // Random chunk size between 5-25 characters
      const chunkSize = Math.floor(Math.random() * 20) + 5;
      const remainingLength = totalLength - position;
      const currentChunkSize = Math.min(chunkSize, remainingLength);
      const chunk = text.substring(position, position + currentChunkSize);
      
      position += currentChunkSize;
      const isLast = position >= totalLength;
      
      onChunk(chunk, isLast);
      
      if (!isLast) {
        // Random delay between 20-80ms
        const delay = Math.floor(Math.random() * 60) + 20;
        setTimeout(streamNextChunk, delay);
      } else {
        onComplete();
      }
    }
  };
  
  streamNextChunk();
}

/**
 * Initialize a markdown streamer with the given element IDs
 * @param containerId ID of the container element
 * @param buttonId ID of the button element to trigger streaming
 * @param progressBarId ID of the progress bar element (optional)
 */
export function initializeStreamer(
  containerId: string, 
  buttonId: string, 
  progressBarId?: string
): void {
  const container = document.getElementById(containerId);
  const button = document.getElementById(buttonId);
  const progressBar = progressBarId ? document.getElementById(progressBarId) : null;
  
  if (!container || !button) {
    console.error('Container or button element not found');
    return;
  }
  
  // Create a new streamer
  const streamer = new MarkdownStreamer(container);
  
  // Add click event to the button
  button.addEventListener('click', () => {
    // Reset the streamer and progress bar
    streamer.reset();
    if (progressBar) {
      progressBar.style.width = '0%';
    }
    
    console.log('Starting streaming process');
    
    const text = sampleMarkdown;
    const totalLength = text.length;
    let processedLength = 0;
    
    // Stream the markdown text
    streamMarkdown(
      text,
      (chunk, isLast) => {
        console.log(`Processing chunk of length ${chunk.length}, isLast: ${isLast}`);
        
        // Process the chunk
        streamer.processChunk(chunk);
        
        // Update progress bar
        if (progressBar) {
          processedLength += chunk.length;
          const progress = (processedLength / totalLength) * 100;
          progressBar.style.width = `${progress}%`;
        }
        
        if (isLast) {
          streamer.finalize();
          console.log('Streaming complete, finalizing');
        }
      },
      () => {
        console.log('Streaming process completed');
      }
    );
  });
}
