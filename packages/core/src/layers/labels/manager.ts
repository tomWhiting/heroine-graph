/**
 * Label Manager
 *
 * Manages label visibility, priority, and collision detection.
 * Handles level-of-detail culling based on zoom level.
 *
 * @module
 */

import type { FontAtlas } from "./atlas.ts";
import { getGlyph, getGlyphUVs, getKerning, measureText } from "./atlas.ts";

/**
 * Position provider for dynamic label positioning
 */
export interface PositionProvider {
  /** Get X position for a node */
  getX(nodeId: number): number;
  /** Get Y position for a node */
  getY(nodeId: number): number;
}

/**
 * Label data for a single node
 */
export interface LabelData {
  /** Node ID */
  nodeId: number;
  /** Label text */
  text: string;
  /** Position in graph space */
  x: number;
  y: number;
  /** Priority (0-1, higher = more important) */
  priority: number;
  /** Minimum zoom level to show this label */
  minZoom?: number;
  /** Maximum zoom level to show this label */
  maxZoom?: number;
}

/**
 * Glyph instance data for GPU rendering
 * Matches the GlyphInstance struct in the vertex shader
 */
export interface GlyphInstance {
  /** Position in graph space (x, y) */
  positionX: number;
  positionY: number;
  /** Glyph size in pixels (width, height) */
  width: number;
  height: number;
  /** UV coordinates in atlas (u0, v0, u1, v1) */
  u0: number;
  v0: number;
  u1: number;
  v1: number;
  /** Offset from baseline (xoffset, yoffset) */
  offsetX: number;
  offsetY: number;
}

/**
 * Bounding box for collision detection
 */
interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Label visibility result
 */
export interface VisibleLabel {
  label: LabelData;
  screenX: number;
  screenY: number;
  width: number;
  height: number;
}

/**
 * Configuration for the label manager
 */
export interface LabelManagerConfig {
  /** Maximum number of visible labels */
  maxLabels: number;
  /** Font size in pixels */
  fontSize: number;
  /** Minimum zoom level to show any labels */
  minZoom: number;
  /** Padding between labels for collision detection */
  labelPadding: number;
  /** Grid cell size for spatial hashing (pixels) */
  gridCellSize: number;
}

const DEFAULT_CONFIG: LabelManagerConfig = {
  maxLabels: 1000,
  fontSize: 14,
  minZoom: 0.3,
  labelPadding: 4,
  gridCellSize: 50,
};

/**
 * Label Manager
 *
 * Handles label visibility culling, priority sorting, and collision detection.
 */
export class LabelManager {
  private config: LabelManagerConfig;
  private labels: LabelData[] = [];
  private fontAtlas: FontAtlas | null = null;

  // Collision grid for spatial hashing
  private occupiedCells: Set<string> = new Set();
  private gridCols: number = 0;
  private gridRows: number = 0;

  constructor(config: Partial<LabelManagerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Set the font atlas for text measurement
   */
  setFontAtlas(atlas: FontAtlas): void {
    this.fontAtlas = atlas;
  }

  /**
   * Update the configuration
   */
  setConfig(config: Partial<LabelManagerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get the total number of labels
   */
  get labelCount(): number {
    return this.labels.length;
  }

  /**
   * Set all labels to manage
   */
  setLabels(labels: LabelData[]): void {
    // Sort by priority (descending) for consistent ordering
    this.labels = [...labels].sort((a, b) => b.priority - a.priority);
  }

  /**
   * Add a single label
   */
  addLabel(label: LabelData): void {
    this.labels.push(label);
    this.labels.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Remove a label by node ID
   */
  removeLabel(nodeId: number): void {
    this.labels = this.labels.filter((l) => l.nodeId !== nodeId);
  }

  /**
   * Clear all labels
   */
  clear(): void {
    this.labels = [];
  }

  /**
   * Get visible labels after culling
   *
   * @param viewportX - Viewport center X in graph space
   * @param viewportY - Viewport center Y in graph space
   * @param scale - Current zoom scale
   * @param canvasWidth - Canvas width in pixels
   * @param canvasHeight - Canvas height in pixels
   * @param positionProvider - Optional provider for dynamic node positions
   * @returns Array of visible labels with screen positions
   */
  getVisibleLabels(
    viewportX: number,
    viewportY: number,
    scale: number,
    canvasWidth: number,
    canvasHeight: number,
    positionProvider?: PositionProvider,
  ): VisibleLabel[] {
    if (!this.fontAtlas) {
      return [];
    }

    // Check minimum zoom threshold
    if (scale < this.config.minZoom) {
      return [];
    }

    // Reset collision grid
    this.initCollisionGrid(canvasWidth, canvasHeight);

    // Calculate viewport bounds in graph space
    const halfWidth = (canvasWidth / 2) / scale;
    const halfHeight = (canvasHeight / 2) / scale;
    const viewLeft = viewportX - halfWidth;
    const viewRight = viewportX + halfWidth;
    const viewTop = viewportY - halfHeight;
    const viewBottom = viewportY + halfHeight;

    const visibleLabels: VisibleLabel[] = [];
    const { fontSize, labelPadding, maxLabels } = this.config;

    for (const label of this.labels) {
      // Stop if we've reached max labels
      if (visibleLabels.length >= maxLabels) {
        break;
      }

      // Check LOD zoom thresholds
      if (label.minZoom !== undefined && scale < label.minZoom) {
        continue;
      }
      if (label.maxZoom !== undefined && scale > label.maxZoom) {
        continue;
      }

      // Get current position - use position provider if available, otherwise use static label position
      const nodeX = positionProvider ? positionProvider.getX(label.nodeId) : label.x;
      const nodeY = positionProvider ? positionProvider.getY(label.nodeId) : label.y;

      // Check if label is in viewport (graph space)
      if (
        nodeX < viewLeft ||
        nodeX > viewRight ||
        nodeY < viewTop ||
        nodeY > viewBottom
      ) {
        continue;
      }

      // Calculate screen position
      const screenX = (nodeX - viewportX) * scale + canvasWidth / 2;
      const screenY = (nodeY - viewportY) * scale + canvasHeight / 2;

      // Measure label dimensions
      const width = measureText(this.fontAtlas, label.text, fontSize);
      const height = fontSize * 1.2; // Approximate line height

      // Center the label horizontally on the node
      const labelX = screenX - width / 2;
      // Position label below the node (with some offset)
      const labelY = screenY + 10;

      // Create bounding box with padding
      const bbox: BoundingBox = {
        x: labelX - labelPadding,
        y: labelY - labelPadding,
        width: width + labelPadding * 2,
        height: height + labelPadding * 2,
      };

      // Check for collision with existing labels
      if (this.checkCollision(bbox)) {
        continue;
      }

      // Mark cells as occupied
      this.markOccupied(bbox);

      visibleLabels.push({
        label,
        screenX: labelX,
        screenY: labelY,
        width,
        height,
      });
    }

    return visibleLabels;
  }

  /**
   * Generate glyph instances for GPU rendering
   *
   * @param visibleLabels - Labels to render
   * @param viewportX - Viewport center X
   * @param viewportY - Viewport center Y
   * @param scale - Current zoom scale
   * @param canvasWidth - Canvas width
   * @param canvasHeight - Canvas height
   * @returns Array of glyph instances and total count
   */
  generateGlyphInstances(
    visibleLabels: VisibleLabel[],
    viewportX: number,
    viewportY: number,
    scale: number,
    canvasWidth: number,
    canvasHeight: number,
  ): { instances: Float32Array; count: number } {
    if (!this.fontAtlas) {
      return { instances: new Float32Array(0), count: 0 };
    }

    const { fontSize } = this.config;
    const atlas = this.fontAtlas;
    const atlasScale = fontSize / atlas.info.size;

    // Estimate total glyphs (overestimate to avoid reallocation)
    let totalGlyphs = 0;
    for (const vl of visibleLabels) {
      totalGlyphs += vl.label.text.length;
    }

    // Each glyph instance: 12 floats (48 bytes) to match WGSL struct alignment
    // GlyphInstance struct in WGSL:
    //   position: vec2<f32> (8 bytes)
    //   size: vec2<f32> (8 bytes)
    //   uv: vec4<f32> (16 bytes)
    //   offset: vec2<f32> (8 bytes)
    //   + 8 bytes padding to align to 16-byte boundary = 48 bytes total
    const instances = new Float32Array(totalGlyphs * 12);
    let instanceIndex = 0;
    let glyphCount = 0;

    for (const vl of visibleLabels) {
      const { label, screenX, screenY } = vl;
      const text = label.text;

      // Convert screen position back to graph position for GPU
      const graphX = (screenX - canvasWidth / 2) / scale + viewportX;
      const graphY = (screenY - canvasHeight / 2) / scale + viewportY;

      let cursorX = 0;
      let prevCharCode: number | null = null;

      for (let i = 0; i < text.length; i++) {
        const charCode = text.charCodeAt(i);
        const glyph = getGlyph(atlas, charCode);

        if (!glyph) {
          // Skip unknown characters
          prevCharCode = charCode;
          continue;
        }

        // Apply kerning
        if (prevCharCode !== null) {
          cursorX += getKerning(atlas, prevCharCode, charCode) * atlasScale;
        }

        // Skip whitespace (no visible glyph)
        if (glyph.width > 0 && glyph.height > 0) {
          const [u0, v0, u1, v1] = getGlyphUVs(atlas, glyph);

          // Position (graph space, but with cursor offset converted back)
          const glyphGraphX = graphX + cursorX / scale;
          const glyphGraphY = graphY;

          instances[instanceIndex++] = glyphGraphX;
          instances[instanceIndex++] = glyphGraphY;
          instances[instanceIndex++] = glyph.width;
          instances[instanceIndex++] = glyph.height;
          instances[instanceIndex++] = u0;
          instances[instanceIndex++] = v0;
          instances[instanceIndex++] = u1;
          instances[instanceIndex++] = v1;
          instances[instanceIndex++] = glyph.xoffset;
          instances[instanceIndex++] = glyph.yoffset;
          instances[instanceIndex++] = 0; // padding for WGSL alignment
          instances[instanceIndex++] = 0; // padding for WGSL alignment

          glyphCount++;
        }

        cursorX += glyph.xadvance * atlasScale;
        prevCharCode = charCode;
      }
    }

    // Return only the used portion
    return {
      instances: instances.subarray(0, instanceIndex),
      count: glyphCount,
    };
  }

  /**
   * Initialize the collision grid
   */
  private initCollisionGrid(canvasWidth: number, canvasHeight: number): void {
    const { gridCellSize } = this.config;
    this.gridCols = Math.ceil(canvasWidth / gridCellSize);
    this.gridRows = Math.ceil(canvasHeight / gridCellSize);
    this.occupiedCells.clear();
  }

  /**
   * Check if a bounding box collides with occupied cells
   */
  private checkCollision(bbox: BoundingBox): boolean {
    const { gridCellSize } = this.config;

    const startCol = Math.max(0, Math.floor(bbox.x / gridCellSize));
    const endCol = Math.min(
      this.gridCols - 1,
      Math.floor((bbox.x + bbox.width) / gridCellSize),
    );
    const startRow = Math.max(0, Math.floor(bbox.y / gridCellSize));
    const endRow = Math.min(
      this.gridRows - 1,
      Math.floor((bbox.y + bbox.height) / gridCellSize),
    );

    for (let row = startRow; row <= endRow; row++) {
      for (let col = startCol; col <= endCol; col++) {
        if (this.occupiedCells.has(`${col},${row}`)) {
          return true;
        }
      }
    }

    return false;
  }

  /**
   * Mark cells as occupied by a bounding box
   */
  private markOccupied(bbox: BoundingBox): void {
    const { gridCellSize } = this.config;

    const startCol = Math.max(0, Math.floor(bbox.x / gridCellSize));
    const endCol = Math.min(
      this.gridCols - 1,
      Math.floor((bbox.x + bbox.width) / gridCellSize),
    );
    const startRow = Math.max(0, Math.floor(bbox.y / gridCellSize));
    const endRow = Math.min(
      this.gridRows - 1,
      Math.floor((bbox.y + bbox.height) / gridCellSize),
    );

    for (let row = startRow; row <= endRow; row++) {
      for (let col = startCol; col <= endCol; col++) {
        this.occupiedCells.add(`${col},${row}`);
      }
    }
  }
}
