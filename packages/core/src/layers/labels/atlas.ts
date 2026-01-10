/**
 * MSDF Font Atlas Loader
 *
 * Loads and manages MSDF (Multi-channel Signed Distance Field) font atlases
 * for high-quality text rendering at any zoom level.
 *
 * Uses BMFont format JSON metadata from msdf-bmfont-xml.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";

/**
 * BMFont character/glyph definition
 */
export interface BMFontChar {
  /** Character code (Unicode) */
  id: number;
  /** X position in atlas texture */
  x: number;
  /** Y position in atlas texture */
  y: number;
  /** Width in atlas texture */
  width: number;
  /** Height in atlas texture */
  height: number;
  /** X offset when rendering */
  xoffset: number;
  /** Y offset when rendering */
  yoffset: number;
  /** How much to advance cursor after this character */
  xadvance: number;
  /** Texture page (always 0 for single-page atlases) */
  page: number;
  /** Color channel (15 = all channels for MSDF) */
  chnl: number;
}

/**
 * BMFont kerning pair
 */
export interface BMFontKerning {
  /** First character code */
  first: number;
  /** Second character code */
  second: number;
  /** Horizontal adjustment */
  amount: number;
}

/**
 * BMFont info section
 */
export interface BMFontInfo {
  /** Font family name */
  face: string;
  /** Font size used for atlas generation */
  size: number;
  /** Bold flag */
  bold: number;
  /** Italic flag */
  italic: number;
  /** Character set */
  charset: string[];
  /** Unicode flag */
  unicode: number;
  /** Stretch height percentage */
  stretchH: number;
  /** Smoothing flag */
  smooth: number;
  /** Anti-aliasing level */
  aa: number;
  /** Padding [top, right, bottom, left] */
  padding: [number, number, number, number];
  /** Spacing [horizontal, vertical] */
  spacing: [number, number];
}

/**
 * BMFont common section
 */
export interface BMFontCommon {
  /** Line height */
  lineHeight: number;
  /** Baseline offset */
  base: number;
  /** Atlas texture width */
  scaleW: number;
  /** Atlas texture height */
  scaleH: number;
  /** Number of texture pages */
  pages: number;
  /** Packed flag */
  packed: number;
  /** Alpha channel content */
  alphaChnl: number;
  /** Red channel content */
  redChnl: number;
  /** Green channel content */
  greenChnl: number;
  /** Blue channel content */
  blueChnl: number;
}

/**
 * Complete BMFont JSON format (from msdf-bmfont-xml)
 */
export interface BMFontData {
  /** Texture page filenames */
  pages: string[];
  /** Character definitions */
  chars: BMFontChar[];
  /** Font metadata */
  info: BMFontInfo;
  /** Common metrics */
  common: BMFontCommon;
  /** Kerning pairs */
  kernings: BMFontKerning[];
}

/**
 * Loaded font atlas with GPU resources
 */
export interface FontAtlas {
  /** Font metadata */
  info: BMFontInfo;
  /** Common metrics */
  common: BMFontCommon;
  /** Character lookup by char code */
  chars: Map<number, BMFontChar>;
  /** Kerning lookup by "first,second" key */
  kernings: Map<string, number>;
  /** Atlas texture */
  texture: GPUTexture;
  /** Texture view for shader binding */
  view: GPUTextureView;
  /** Texture sampler */
  sampler: GPUSampler;
  /** MSDF distance range (pixels) - must match atlas generation */
  distanceRange: number;
  /** Destroy GPU resources */
  destroy: () => void;
}

/**
 * Options for loading a font atlas
 */
export interface FontAtlasOptions {
  /** MSDF distance range used during atlas generation (default: 4) */
  distanceRange?: number;
}

/**
 * Load a font atlas from JSON metadata and PNG texture
 *
 * @param context - WebGPU context
 * @param jsonUrl - URL to BMFont JSON file
 * @param pngUrl - URL to atlas PNG texture
 * @param options - Loading options
 * @returns Loaded font atlas with GPU resources
 */
export async function loadFontAtlas(
  context: GPUContext,
  jsonUrl: string,
  pngUrl: string,
  options: FontAtlasOptions = {}
): Promise<FontAtlas> {
  const { device } = context;
  const { distanceRange = 4 } = options;

  // Load JSON metadata
  const jsonResponse = await fetch(jsonUrl);
  if (!jsonResponse.ok) {
    throw new Error(`Failed to load font metadata: ${jsonResponse.status} ${jsonUrl}`);
  }
  const fontData: BMFontData = await jsonResponse.json();

  // Load PNG texture via canvas to get pixel data
  const pngResponse = await fetch(pngUrl);
  if (!pngResponse.ok) {
    throw new Error(`Failed to load font texture: ${pngResponse.status} ${pngUrl}`);
  }
  const pngBlob = await pngResponse.blob();
  const imageBitmap = await createImageBitmap(pngBlob);

  // Use OffscreenCanvas to extract pixel data
  const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create 2D context for font atlas");
  }
  ctx.drawImage(imageBitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, imageBitmap.width, imageBitmap.height);

  // Create GPU texture
  const texture = device.createTexture({
    label: `Font Atlas Texture (${fontData.info.face})`,
    size: {
      width: imageBitmap.width,
      height: imageBitmap.height,
    },
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Upload image data using writeTexture
  device.queue.writeTexture(
    { texture },
    imageData.data,
    { bytesPerRow: imageBitmap.width * 4 },
    { width: imageBitmap.width, height: imageBitmap.height }
  );

  // Create texture view
  const view = texture.createView({
    label: `Font Atlas View (${fontData.info.face})`,
  });

  // Create sampler with linear filtering (important for SDF quality)
  const sampler = device.createSampler({
    label: `Font Atlas Sampler (${fontData.info.face})`,
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  // Build character lookup map
  const chars = new Map<number, BMFontChar>();
  for (const char of fontData.chars) {
    chars.set(char.id, char);
  }

  // Build kerning lookup map
  const kernings = new Map<string, number>();
  for (const kern of fontData.kernings) {
    kernings.set(`${kern.first},${kern.second}`, kern.amount);
  }

  // Font atlas loaded successfully

  return {
    info: fontData.info,
    common: fontData.common,
    chars,
    kernings,
    texture,
    view,
    sampler,
    distanceRange,
    destroy: () => {
      texture.destroy();
    },
  };
}

/**
 * Get glyph data for a character
 *
 * @param atlas - Loaded font atlas
 * @param charCode - Unicode character code
 * @returns Glyph data or undefined if not found
 */
export function getGlyph(atlas: FontAtlas, charCode: number): BMFontChar | undefined {
  return atlas.chars.get(charCode);
}

/**
 * Get kerning adjustment between two characters
 *
 * @param atlas - Loaded font atlas
 * @param first - First character code
 * @param second - Second character code
 * @returns Kerning amount (0 if no kerning defined)
 */
export function getKerning(atlas: FontAtlas, first: number, second: number): number {
  return atlas.kernings.get(`${first},${second}`) ?? 0;
}

/**
 * Measure text width using the font atlas
 *
 * @param atlas - Loaded font atlas
 * @param text - Text to measure
 * @param fontSize - Desired font size
 * @returns Width in pixels at the given font size
 */
export function measureText(atlas: FontAtlas, text: string, fontSize: number): number {
  const scale = fontSize / atlas.info.size;
  let width = 0;
  let prevCharCode: number | null = null;

  for (let i = 0; i < text.length; i++) {
    const charCode = text.charCodeAt(i);
    const glyph = atlas.chars.get(charCode);

    if (glyph) {
      // Add kerning if applicable
      if (prevCharCode !== null) {
        width += getKerning(atlas, prevCharCode, charCode) * scale;
      }
      width += glyph.xadvance * scale;
    }

    prevCharCode = charCode;
  }

  return width;
}

/**
 * Calculate UV coordinates for a glyph in the atlas
 *
 * @param atlas - Loaded font atlas
 * @param glyph - Glyph to calculate UVs for
 * @returns UV coordinates [u0, v0, u1, v1]
 */
export function getGlyphUVs(
  atlas: FontAtlas,
  glyph: BMFontChar
): [number, number, number, number] {
  const { scaleW, scaleH } = atlas.common;

  const u0 = glyph.x / scaleW;
  const v0 = glyph.y / scaleH;
  const u1 = (glyph.x + glyph.width) / scaleW;
  const v1 = (glyph.y + glyph.height) / scaleH;

  return [u0, v0, u1, v1];
}

/**
 * Default font atlas paths (Roboto MSDF)
 * These paths are relative to where the bundle is served from.
 * For Storybook, assets are served from dist/assets/fonts/
 */
export const DEFAULT_FONT_ATLAS_JSON = "./assets/fonts/roboto-msdf.json";
export const DEFAULT_FONT_ATLAS_PNG = "./assets/fonts/roboto-msdf.png";

/**
 * Load the default Roboto MSDF font atlas
 */
export async function loadDefaultFontAtlas(context: GPUContext): Promise<FontAtlas> {
  return loadFontAtlas(context, DEFAULT_FONT_ATLAS_JSON, DEFAULT_FONT_ATLAS_PNG);
}
