/**
 * Error Handling for Heroine Graph
 *
 * Per Constitution V: No Silent Failures.
 * All errors include context and are actionable.
 */

/**
 * Error codes for categorizing errors.
 */
export enum ErrorCode {
  // Initialization errors (1xxx)
  WEBGPU_NOT_SUPPORTED = 1001,
  WEBGPU_ADAPTER_FAILED = 1002,
  WEBGPU_DEVICE_FAILED = 1003,
  WASM_LOAD_FAILED = 1004,
  CANVAS_NOT_FOUND = 1005,

  // Shader errors (2xxx)
  SHADER_COMPILATION_FAILED = 2001,
  SHADER_NOT_FOUND = 2002,
  PIPELINE_CREATION_FAILED = 2003,

  // Data errors (3xxx)
  INVALID_NODE_ID = 3001,
  INVALID_EDGE_ID = 3002,
  NODE_NOT_FOUND = 3003,
  EDGE_NOT_FOUND = 3004,
  INVALID_GRAPH_DATA = 3005,
  INVALID_POSITIONS = 3006,

  // Buffer errors (4xxx)
  BUFFER_CREATION_FAILED = 4001,
  BUFFER_WRITE_FAILED = 4002,
  TEXTURE_CREATION_FAILED = 4003,

  // Render errors (5xxx)
  RENDER_PASS_FAILED = 5001,
  COMMAND_ENCODING_FAILED = 5002,

  // Layer errors (6xxx)
  LAYER_NOT_FOUND = 6001,
  INVALID_LAYER_CONFIG = 6002,

  // Lifecycle errors (7xxx)
  DISPOSED_ACCESS = 7001,

  // General errors (9xxx)
  UNKNOWN_ERROR = 9999,
}

/**
 * Error context for debugging.
 */
export interface ErrorContext {
  /** Original error if wrapping */
  readonly originalError?: unknown | undefined;
  /** Shader source line if shader error */
  readonly shaderLine?: number | undefined;
  /** Shader source snippet if shader error */
  readonly shaderSnippet?: string | undefined;
  /** Node ID if node-related error */
  readonly nodeId?: number | undefined;
  /** Edge ID if edge-related error */
  readonly edgeId?: number | undefined;
  /** Additional context data */
  readonly [key: string]: unknown;
}

/**
 * Custom error class for Heroine Graph.
 *
 * Provides structured error information with context for debugging.
 */
export class HeroineGraphError extends Error {
  /** Error code for categorization */
  readonly code: ErrorCode;
  /** Additional context for debugging */
  readonly context: ErrorContext;
  /** Suggested fix if available */
  readonly suggestion?: string | undefined;

  constructor(
    code: ErrorCode,
    message: string,
    context: ErrorContext = {},
    suggestion?: string
  ) {
    super(message);
    this.name = "HeroineGraphError";
    this.code = code;
    this.context = context;
    this.suggestion = suggestion;

    // Maintains proper stack trace for where error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, HeroineGraphError);
    }
  }

  /**
   * Create a formatted error message with context.
   */
  toDetailedString(): string {
    const parts: string[] = [
      `[HeroineGraph Error ${this.code}] ${this.message}`,
    ];

    if (this.suggestion) {
      parts.push(`Suggestion: ${this.suggestion}`);
    }

    if (Object.keys(this.context).length > 0) {
      parts.push("Context:");
      for (const [key, value] of Object.entries(this.context)) {
        if (key === "originalError" && value instanceof Error) {
          parts.push(`  ${key}: ${value.message}`);
        } else if (value !== undefined) {
          parts.push(`  ${key}: ${JSON.stringify(value)}`);
        }
      }
    }

    return parts.join("\n");
  }

  /**
   * Log the error with full context.
   */
  log(): void {
    console.error(this.toDetailedString());
    if (this.stack) {
      console.error(this.stack);
    }
  }
}

/**
 * Error factory functions for common error types.
 */
export const Errors = {
  /**
   * Create a WebGPU not supported error.
   */
  webgpuNotSupported(reason: string): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.WEBGPU_NOT_SUPPORTED,
      `WebGPU is not supported: ${reason}`,
      {},
      "Use Chrome 113+, Firefox 141+, Safari 26+, or Edge 113+. " +
        "Ensure hardware acceleration is enabled in browser settings."
    );
  },

  /**
   * Create a WebGPU adapter request failed error.
   */
  adapterFailed(): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.WEBGPU_ADAPTER_FAILED,
      "Failed to request WebGPU adapter",
      {},
      "Check GPU driver installation. Try updating graphics drivers. " +
        "Some integrated GPUs may not support WebGPU."
    );
  },

  /**
   * Create a WebGPU device request failed error.
   */
  deviceFailed(limits?: string[]): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.WEBGPU_DEVICE_FAILED,
      "Failed to request WebGPU device",
      { unsupportedLimits: limits },
      "The GPU may not support required features. " +
        "Try reducing graph size or disabling advanced features."
    );
  },

  /**
   * Create a canvas not found error.
   */
  canvasNotFound(selector: string): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.CANVAS_NOT_FOUND,
      `Canvas element not found: "${selector}"`,
      { selector },
      "Ensure the canvas element exists in the DOM before creating the graph."
    );
  },

  /**
   * Create a shader compilation error.
   */
  shaderCompilationFailed(
    shaderName: string,
    messages: string,
    line?: number,
    snippet?: string
  ): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.SHADER_COMPILATION_FAILED,
      `Shader compilation failed: ${shaderName}`,
      {
        shaderName,
        messages,
        shaderLine: line,
        shaderSnippet: snippet,
      },
      "This is a library bug. Please report it with the error details."
    );
  },

  /**
   * Create a node not found error.
   */
  nodeNotFound(nodeId: number): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.NODE_NOT_FOUND,
      `Node not found: ${nodeId}`,
      { nodeId },
      "Ensure the node exists before performing operations on it."
    );
  },

  /**
   * Create an edge not found error.
   */
  edgeNotFound(edgeId: number): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.EDGE_NOT_FOUND,
      `Edge not found: ${edgeId}`,
      { edgeId },
      "Ensure the edge exists before performing operations on it."
    );
  },

  /**
   * Create an invalid graph data error.
   */
  invalidGraphData(reason: string): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.INVALID_GRAPH_DATA,
      `Invalid graph data: ${reason}`,
      {},
      "Check that all nodes have unique IDs and all edges reference existing nodes."
    );
  },

  /**
   * Create a layer not found error.
   */
  layerNotFound(layerType: string): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.LAYER_NOT_FOUND,
      `Layer not found: ${layerType}`,
      { layerType },
      "Valid layer types are: heatmap, contour, metaball, labels."
    );
  },

  /**
   * Create a buffer creation failed error.
   */
  bufferCreationFailed(bufferName: string, reason: string): HeroineGraphError {
    return new HeroineGraphError(
      ErrorCode.BUFFER_CREATION_FAILED,
      `Failed to create GPU buffer "${bufferName}": ${reason}`,
      { bufferName },
      "The GPU may be out of memory. Try reducing graph size."
    );
  },
};

/**
 * Assert a condition is true, throwing if false.
 *
 * @param condition Condition to check
 * @param error Error to throw if condition is false
 */
export function assert(
  condition: boolean,
  error: HeroineGraphError
): asserts condition {
  if (!condition) {
    throw error;
  }
}

/**
 * Wrap an async function to convert unknown errors to HeroineGraphError.
 */
export async function wrapAsync<T>(
  fn: () => Promise<T>,
  code: ErrorCode,
  message: string
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (error instanceof HeroineGraphError) {
      throw error;
    }
    throw new HeroineGraphError(code, message, { originalError: error });
  }
}
