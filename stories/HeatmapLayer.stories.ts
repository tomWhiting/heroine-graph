import type { Meta, StoryObj } from '@storybook/html';
import './heroine-graph.css';
import { generateClusteredGraph, generateSpiralGraph, type GraphData } from './graph-generators.ts';

interface HeatmapConfig {
  colorScale: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'turbo';
  radius: number;
  intensity: number;
  opacity: number;
  enabled: boolean;
}

interface HeroineGraph {
  frameStats: { fps?: number; avgFrameTime?: number } | null;
  nodeCount: number;
  resize: (width: number, height: number) => void;
  load: (data: GraphData) => Promise<void>;
  startSimulation: () => void;
  pauseSimulation: () => void;
  fitToView: () => void;
  pan: (dx: number, dy: number) => void;
  zoom: (factor: number) => void;
  enableHeatmap?: (config?: Partial<HeatmapConfig>) => void;
  disableHeatmap?: () => void;
  setHeatmapConfig?: (config: Partial<HeatmapConfig>) => void;
}

interface HeroineGraphModule {
  createHeroineGraph: (options: { canvas: HTMLCanvasElement; debug?: boolean }) => Promise<HeroineGraph>;
  getSupportInfo: () => Promise<{ supported: boolean; reason?: string }>;
}

const meta: Meta<HeatmapConfig> = {
  title: 'HeroineGraph/Heatmap Layer',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'Heatmap visualization layer showing node density as a color gradient overlay.',
      },
    },
  },
  argTypes: {
    colorScale: {
      control: { type: 'select' },
      options: ['viridis', 'plasma', 'inferno', 'magma', 'turbo'],
      description: 'Color scale for the heatmap',
    },
    radius: {
      control: { type: 'range', min: 10, max: 150, step: 5 },
      description: 'Gaussian splat radius',
    },
    intensity: {
      control: { type: 'range', min: 0.01, max: 0.5, step: 0.01 },
      description: 'Intensity per node',
    },
    opacity: {
      control: { type: 'range', min: 0.1, max: 1.0, step: 0.1 },
      description: 'Layer opacity',
    },
    enabled: {
      control: 'boolean',
      description: 'Enable/disable heatmap layer',
    },
  },
};

export default meta;
type Story = StoryObj<HeatmapConfig>;

/**
 * Create the heatmap graph container with controls
 */
function createHeatmapContainer(title: string, initialConfig: HeatmapConfig): HTMLDivElement {
  const container = document.createElement('div');
  container.className = 'heroine-graph-container';
  container.innerHTML = `
    <div class="heroine-graph-header">
      <h3 class="heroine-graph-title">${title}</h3>
      <div class="heroine-graph-stats">
        <div class="heroine-graph-stat">
          <span class="heroine-graph-stat-label">FPS:</span>
          <span class="heroine-graph-stat-value" data-stat="fps">--</span>
        </div>
        <div class="heroine-graph-stat">
          <span class="heroine-graph-stat-label">Nodes:</span>
          <span class="heroine-graph-stat-value" data-stat="nodes">--</span>
        </div>
      </div>
    </div>
    <canvas data-graph-canvas></canvas>
    <div class="heroine-graph-loading">
      <div class="heroine-graph-spinner"></div>
      <div data-loading-text>Initializing HeroineGraph...</div>
    </div>
    <div class="heroine-graph-heatmap-controls" style="display: none;">
      <h4>Heatmap Controls</h4>
      <div class="heroine-graph-control-row">
        <label>Enabled</label>
        <input type="checkbox" data-heatmap="enabled" ${initialConfig.enabled ? 'checked' : ''}>
      </div>
      <div class="heroine-graph-control-row">
        <label>Color Scale</label>
        <select data-heatmap="colorScale">
          <option value="viridis" ${initialConfig.colorScale === 'viridis' ? 'selected' : ''}>Viridis</option>
          <option value="plasma" ${initialConfig.colorScale === 'plasma' ? 'selected' : ''}>Plasma</option>
          <option value="inferno" ${initialConfig.colorScale === 'inferno' ? 'selected' : ''}>Inferno</option>
          <option value="magma" ${initialConfig.colorScale === 'magma' ? 'selected' : ''}>Magma</option>
          <option value="turbo" ${initialConfig.colorScale === 'turbo' ? 'selected' : ''}>Turbo</option>
        </select>
      </div>
      <div class="heroine-graph-control-row">
        <label>Radius: <span data-value="radius">${initialConfig.radius}</span></label>
        <input type="range" data-heatmap="radius" min="10" max="150" step="5" value="${initialConfig.radius}">
      </div>
      <div class="heroine-graph-control-row">
        <label>Intensity: <span data-value="intensity">${initialConfig.intensity}</span></label>
        <input type="range" data-heatmap="intensity" min="0.01" max="0.5" step="0.01" value="${initialConfig.intensity}">
      </div>
      <div class="heroine-graph-control-row">
        <label>Opacity: <span data-value="opacity">${initialConfig.opacity}</span></label>
        <input type="range" data-heatmap="opacity" min="0.1" max="1.0" step="0.1" value="${initialConfig.opacity}">
      </div>
    </div>
    <div class="heroine-graph-controls" style="display: none;">
      <button class="heroine-graph-btn secondary" data-action="toggle-sim">Pause</button>
      <button class="heroine-graph-btn secondary" data-action="fit-view">Fit View</button>
    </div>
  `;
  return container;
}

/**
 * Wait for container to be in DOM and have valid dimensions
 */
function waitForLayout(container: HTMLElement, maxWait = 5000): Promise<void> {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    const check = () => {
      // Check if element is connected to DOM and has dimensions
      if (container.isConnected) {
        const rect = container.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          resolve();
          return;
        }
      }
      // Timeout check
      if (Date.now() - startTime > maxWait) {
        reject(new Error('Timeout waiting for layout'));
        return;
      }
      requestAnimationFrame(check);
    };
    requestAnimationFrame(check);
  });
}

/**
 * Initialize the graph with heatmap layer
 */
async function initHeatmapGraph(
  container: HTMLElement,
  graphData: GraphData,
  heatmapConfig: HeatmapConfig
): Promise<HeroineGraph | undefined> {
  const canvas = container.querySelector<HTMLCanvasElement>('[data-graph-canvas]')!;
  const loading = container.querySelector<HTMLElement>('.heroine-graph-loading')!;
  const loadingText = container.querySelector<HTMLElement>('[data-loading-text]')!;
  const controls = container.querySelector<HTMLElement>('.heroine-graph-controls')!;
  const heatmapControls = container.querySelector<HTMLElement>('.heroine-graph-heatmap-controls')!;

  try {
    // Wait for container to be laid out with valid dimensions
    await waitForLayout(container);

    // Size canvas BEFORE creating graph context
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;

    loadingText.textContent = 'Loading HeroineGraph...';
    const { createHeroineGraph, getSupportInfo } = await import('../dist/heroine-graph.esm.js') as HeroineGraphModule;

    loadingText.textContent = 'Checking WebGPU support...';
    const support = await getSupportInfo();

    if (!support.supported) {
      throw new Error(support.reason || 'WebGPU is not supported in this browser');
    }

    loadingText.textContent = 'Initializing WebGPU...';
    const graph = await createHeroineGraph({ canvas, debug: false });

    // Handle resize
    const resizeCanvas = () => {
      const r = container.getBoundingClientRect();
      canvas.width = r.width * window.devicePixelRatio;
      canvas.height = r.height * window.devicePixelRatio;
      graph.resize(canvas.width, canvas.height);
    };
    window.addEventListener('resize', resizeCanvas);

    // Load data
    loadingText.textContent = 'Loading graph data...';
    await graph.load(graphData);

    // Enable heatmap layer
    if (graph.enableHeatmap) {
      graph.enableHeatmap(heatmapConfig);
    }

    // Hide loading, show controls
    loading.style.display = 'none';
    controls.style.display = 'flex';
    heatmapControls.style.display = 'block';

    // Update stats
    const fpsEl = container.querySelector<HTMLElement>('[data-stat="fps"]');
    const nodesEl = container.querySelector<HTMLElement>('[data-stat="nodes"]');

    if (fpsEl) {
      setInterval(() => {
        const stats = graph.frameStats;
        if (stats) {
          fpsEl.textContent = stats.fps?.toFixed(0) || '--';
        }
        if (nodesEl) nodesEl.textContent = graph.nodeCount?.toLocaleString() || '--';
      }, 100);
    }

    // Heatmap controls
    const enabledCheckbox = container.querySelector<HTMLInputElement>('[data-heatmap="enabled"]');
    const colorScaleSelect = container.querySelector<HTMLSelectElement>('[data-heatmap="colorScale"]');
    const radiusSlider = container.querySelector<HTMLInputElement>('[data-heatmap="radius"]');
    const intensitySlider = container.querySelector<HTMLInputElement>('[data-heatmap="intensity"]');
    const opacitySlider = container.querySelector<HTMLInputElement>('[data-heatmap="opacity"]');

    enabledCheckbox?.addEventListener('change', (e) => {
      const target = e.target as HTMLInputElement;
      if (target.checked) {
        graph.enableHeatmap?.();
      } else {
        graph.disableHeatmap?.();
      }
    });

    colorScaleSelect?.addEventListener('change', (e) => {
      const target = e.target as HTMLSelectElement;
      graph.setHeatmapConfig?.({ colorScale: target.value as HeatmapConfig['colorScale'] });
    });

    radiusSlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="radius"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setHeatmapConfig?.({ radius: value });
    });

    intensitySlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="intensity"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setHeatmapConfig?.({ intensity: value });
    });

    opacitySlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="opacity"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setHeatmapConfig?.({ opacity: value });
    });

    // Simulation controls
    let simPaused = false;
    const toggleSimBtn = container.querySelector<HTMLButtonElement>('[data-action="toggle-sim"]');
    toggleSimBtn?.addEventListener('click', () => {
      if (simPaused) {
        graph.startSimulation();
        toggleSimBtn.textContent = 'Pause';
      } else {
        graph.pauseSimulation();
        toggleSimBtn.textContent = 'Resume';
      }
      simPaused = !simPaused;
    });

    container.querySelector('[data-action="fit-view"]')?.addEventListener('click', () => {
      graph.fitToView();
    });

    // Pan/zoom
    let isDragging = false;
    let lastMouse = { x: 0, y: 0 };

    canvas.addEventListener('mousedown', (e: MouseEvent) => {
      isDragging = true;
      lastMouse = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mousemove', (e: MouseEvent) => {
      if (isDragging) {
        graph.pan(e.clientX - lastMouse.x, e.clientY - lastMouse.y);
        lastMouse = { x: e.clientX, y: e.clientY };
      }
    });

    canvas.addEventListener('mouseup', () => { isDragging = false; });
    canvas.addEventListener('mouseleave', () => { isDragging = false; });

    canvas.addEventListener('wheel', (e: WheelEvent) => {
      e.preventDefault();
      graph.zoom(e.deltaY > 0 ? 0.9 : 1.1);
    }, { passive: false });

    return graph;
  } catch (err) {
    const error = err as Error;
    loading.innerHTML = `
      <div class="heroine-graph-error">
        <h3>Initialization Error</h3>
        <p>${error.message}</p>
        <p style="margin-top: 12px; font-size: 12px;">
          WebGPU requires Chrome 113+, Edge 113+, or Safari 18+
        </p>
      </div>
    `;
    console.error('HeroineGraph initialization failed:', err);
  }
}

/**
 * Clustered Graph with Heatmap - Shows density in cluster regions
 */
export const ClusteredGraph: Story = {
  args: {
    colorScale: 'viridis',
    radius: 60,
    intensity: 0.15,
    opacity: 0.7,
    enabled: true,
  },
  render: (args) => {
    const container = createHeatmapContainer('Clustered Graph with Density Heatmap', args);
    container.dataset.heatmapConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.heatmapConfig || '{}') as HeatmapConfig;
    const data = generateClusteredGraph(5, 100);
    await initHeatmapGraph(container, data, config);
  },
};

/**
 * Spiral Galaxy with Heatmap - Beautiful spiral pattern visualization
 */
export const SpiralGalaxy: Story = {
  args: {
    colorScale: 'plasma',
    radius: 40,
    intensity: 0.1,
    opacity: 0.8,
    enabled: true,
  },
  render: (args) => {
    const container = createHeatmapContainer('Spiral Galaxy with Density Heatmap', args);
    container.dataset.heatmapConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.heatmapConfig || '{}') as HeatmapConfig;
    const data = generateSpiralGraph(1500, 4);
    await initHeatmapGraph(container, data, config);
  },
};

/**
 * Color Scale Comparison - Try different color scales
 */
export const ColorScaleComparison: Story = {
  args: {
    colorScale: 'inferno',
    radius: 50,
    intensity: 0.12,
    opacity: 0.75,
    enabled: true,
  },
  render: (args) => {
    const container = createHeatmapContainer('Color Scale Comparison', args);
    container.dataset.heatmapConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.heatmapConfig || '{}') as HeatmapConfig;
    const data = generateClusteredGraph(7, 80);
    await initHeatmapGraph(container, data, config);
  },
};

/**
 * High Density - Large clustered graph for stress testing heatmap
 */
export const HighDensity: Story = {
  args: {
    colorScale: 'turbo',
    radius: 30,
    intensity: 0.08,
    opacity: 0.65,
    enabled: true,
  },
  render: (args) => {
    const container = createHeatmapContainer('High Density Heatmap (3000 nodes)', args);
    container.dataset.heatmapConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.heatmapConfig || '{}') as HeatmapConfig;
    const data = generateClusteredGraph(10, 300);
    await initHeatmapGraph(container, data, config);
  },
};
