/**
 * Heatmap Density Texture
 *
 * Creates and manages the render target texture for density accumulation.
 * Gaussian splats are rendered additively to this texture to build up
 * the density field before color mapping.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";

/**
 * Density texture configuration
 */
export interface DensityTextureConfig {
  /** Texture width (usually matches canvas width) */
  width: number;
  /** Texture height (usually matches canvas height) */
  height: number;
  /** Resolution scale (0.5 = half resolution for performance) */
  scale?: number;
}

/**
 * Density texture resources
 */
export interface DensityTexture {
  /** The density texture (RGBA16Float for HDR accumulation) */
  texture: GPUTexture;
  /** View for rendering to the texture */
  renderView: GPUTextureView;
  /** View for sampling in the colormap pass */
  sampleView: GPUTextureView;
  /** Sampler for color mapping */
  sampler: GPUSampler;
  /** Current width */
  width: number;
  /** Current height */
  height: number;
  /** Destroy the texture resources */
  destroy: () => void;
  /** Resize the texture */
  resize: (width: number, height: number) => void;
}

/**
 * Default density texture configuration
 */
export const DEFAULT_DENSITY_TEXTURE_CONFIG: Required<DensityTextureConfig> = {
  width: 800,
  height: 600,
  scale: 1.0,
};

/**
 * Creates a density texture for heatmap accumulation.
 *
 * The texture uses RGBA16Float format for HDR accumulation, allowing
 * many overlapping splats without clamping.
 *
 * @param context GPU context
 * @param config Texture configuration
 * @returns Density texture resources
 */
export function createDensityTexture(
  context: GPUContext,
  config?: Partial<DensityTextureConfig>,
): DensityTexture {
  const { device } = context;
  const finalConfig = { ...DEFAULT_DENSITY_TEXTURE_CONFIG, ...config };

  const scaledWidth = Math.max(1, Math.floor(finalConfig.width * finalConfig.scale));
  const scaledHeight = Math.max(1, Math.floor(finalConfig.height * finalConfig.scale));

  // Create the density texture
  let texture = device.createTexture({
    label: "Heatmap Density Texture",
    size: { width: scaledWidth, height: scaledHeight },
    format: "rgba16float",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING,
  });

  // Create views
  let renderView = texture.createView({
    label: "Density Render View",
  });

  let sampleView = texture.createView({
    label: "Density Sample View",
  });

  // Create sampler for color mapping pass
  const sampler = device.createSampler({
    label: "Density Sampler",
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  let currentWidth = scaledWidth;
  let currentHeight = scaledHeight;

  function destroy(): void {
    texture.destroy();
  }

  function resize(width: number, height: number): void {
    const newWidth = Math.max(1, Math.floor(width * finalConfig.scale));
    const newHeight = Math.max(1, Math.floor(height * finalConfig.scale));

    if (newWidth === currentWidth && newHeight === currentHeight) {
      return;
    }

    // Destroy old texture
    texture.destroy();

    // Create new texture
    texture = device.createTexture({
      label: "Heatmap Density Texture",
      size: { width: newWidth, height: newHeight },
      format: "rgba16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING,
    });

    renderView = texture.createView({
      label: "Density Render View",
    });

    sampleView = texture.createView({
      label: "Density Sample View",
    });

    currentWidth = newWidth;
    currentHeight = newHeight;
  }

  return {
    get texture() { return texture; },
    get renderView() { return renderView; },
    get sampleView() { return sampleView; },
    sampler,
    get width() { return currentWidth; },
    get height() { return currentHeight; },
    destroy,
    resize,
  };
}

/**
 * Clear the density texture to zero.
 *
 * Call this at the start of each frame before rendering splats.
 *
 * @param encoder Command encoder
 * @param densityTexture Density texture to clear
 */
export function clearDensityTexture(
  encoder: GPUCommandEncoder,
  densityTexture: DensityTexture,
): void {
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: densityTexture.renderView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass.end();
}
