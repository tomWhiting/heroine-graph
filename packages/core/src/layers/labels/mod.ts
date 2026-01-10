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
export { type LabelData, LabelManager, type VisibleLabel } from "./manager.ts";
export { DEFAULT_LABEL_CONFIG, type LabelConfig } from "./config.ts";
export {
  type BMFontChar,
  type BMFontData,
  type FontAtlas,
  getGlyph,
  getGlyphUVs,
  getKerning,
  loadDefaultFontAtlas,
  loadFontAtlas,
  measureText,
} from "./atlas.ts";
