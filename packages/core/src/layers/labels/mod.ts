/**
 * Labels Layer Module
 *
 * Provides MSDF text rendering for node labels with:
 * - Priority-based label selection
 * - Collision detection to prevent overlap
 * - Level-of-detail culling based on zoom
 * - Sharp text at any zoom level
 *
 * @module
 */

export { LabelsLayer, type LabelsRenderContext, type PositionProvider } from "./layer.ts";
export { LabelManager, type LabelData, type VisibleLabel } from "./manager.ts";
export {
  type LabelConfig,
  DEFAULT_LABEL_CONFIG,
} from "./config.ts";
export {
  type FontAtlas,
  type BMFontChar,
  type BMFontData,
  loadFontAtlas,
  loadDefaultFontAtlas,
  getGlyph,
  getKerning,
  getGlyphUVs,
  measureText,
} from "./atlas.ts";
