import type { Meta, StoryObj } from '@storybook/html';
import './heroine-graph.css';
import { generateClusteredGraph, generateSpiralGraph, type GraphData } from './graph-generators.ts';

interface ContourConfig {
  thresholds: number[];
  strokeWidth: number;
  strokeColor: string;
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
  enableContour?: (config?: Partial<ContourConfig>) => void;
  disableContour?: () => void;
  setContourConfig?: (config: Partial<ContourConfig>) => void;
  enableHeatmap?: (config?: Record<string, unknown>) => void;
}

interface HeroineGraphModule {
  createHeroineGraph: (options: { canvas: HTMLCanvasElement; debug?: boolean }) => Promise<HeroineGraph>;
  getSupportInfo: () => Promise<{ supported: boolean; reason?: string }>;
}

const meta: Meta<ContourConfig> = {
  title: 'HeroineGraph/Contour Layer',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'Contour visualization layer rendering iso-lines at specified density thresholds using marching squares algorithm.',
      },
    },
  },
  argTypes: {
    strokeWidth: {
      control: { type: 'range', min: 0.5, max: 5, step: 0.5 },
      description: 'Contour line width',
    },
    strokeColor: {
      control: 'color',
      description: 'Contour line color',
    },
    opacity: {
      control: { type: 'range', min: 0.1, max: 1.0, step: 0.1 },
      description: 'Layer opacity',
    },
    enabled: {
      control: 'boolean',
      description: 'Enable/disable contour layer',
    },
  },
};

export default meta;
type Story = StoryObj<ContourConfig>;

/**
 * Create the contour graph container with controls
 */
function createContourContainer(title: string, initialConfig: ContourConfig): HTMLDivElement {
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
    <div class="heroine-graph-contour-controls" style="display: none;">
      <h4>Contour Controls</h4>
      <div class="heroine-graph-control-row">
        <label>Enabled</label>
        <input type="checkbox" data-contour="enabled" ${initialConfig.enabled ? 'checked' : ''}>
      </div>
      <div class="heroine-graph-control-row">
        <label>Stroke Width: <span data-value="strokeWidth">${initialConfig.strokeWidth}</span></label>
        <input type="range" data-contour="strokeWidth" min="0.5" max="5" step="0.5" value="${initialConfig.strokeWidth}">
      </div>
      <div class="heroine-graph-control-row">
        <label>Stroke Color</label>
        <input type="color" data-contour="strokeColor" value="${initialConfig.strokeColor}">
      </div>
      <div class="heroine-graph-control-row">
        <label>Opacity: <span data-value="opacity">${initialConfig.opacity}</span></label>
        <input type="range" data-contour="opacity" min="0.1" max="1.0" step="0.1" value="${initialConfig.opacity}">
      </div>
      <div class="heroine-graph-control-row">
        <label>Thresholds</label>
        <select data-contour="thresholds">
          <option value="single">Single (0.5)</option>
          <option value="few">Few (0.3, 0.5, 0.7)</option>
          <option value="many">Many (0.2, 0.4, 0.6, 0.8)</option>
          <option value="dense">Dense (0.1-0.9)</option>
        </select>
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
      if (container.isConnected) {
        const rect = container.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          resolve();
          return;
        }
      }
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
 * Parse thresholds from preset name
 */
function parseThresholds(preset: string): number[] {
  switch (preset) {
    case 'single': return [0.5];
    case 'few': return [0.3, 0.5, 0.7];
    case 'many': return [0.2, 0.4, 0.6, 0.8];
    case 'dense': return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    default: return [0.5];
  }
}

/**
 * Initialize the graph with contour layer
 */
async function initContourGraph(
  container: HTMLElement,
  graphData: GraphData,
  contourConfig: ContourConfig
): Promise<HeroineGraph | undefined> {
  const canvas = container.querySelector<HTMLCanvasElement>('[data-graph-canvas]')!;
  const loading = container.querySelector<HTMLElement>('.heroine-graph-loading')!;
  const loadingText = container.querySelector<HTMLElement>('[data-loading-text]')!;
  const controls = container.querySelector<HTMLElement>('.heroine-graph-controls')!;
  const contourControls = container.querySelector<HTMLElement>('.heroine-graph-contour-controls')!;

  try {
    await waitForLayout(container);

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

    const resizeCanvas = () => {
      const r = container.getBoundingClientRect();
      canvas.width = r.width * window.devicePixelRatio;
      canvas.height = r.height * window.devicePixelRatio;
      graph.resize(canvas.width, canvas.height);
    };
    window.addEventListener('resize', resizeCanvas);

    loadingText.textContent = 'Loading graph data...';
    await graph.load(graphData);

    // Enable heatmap first (contours use density texture from heatmap)
    if (graph.enableHeatmap) {
      graph.enableHeatmap({ opacity: 0.3, radius: 50, intensity: 0.1 });
    }

    // Enable contour layer
    if (graph.enableContour) {
      graph.enableContour(contourConfig);
    }

    loading.style.display = 'none';
    controls.style.display = 'flex';
    contourControls.style.display = 'block';

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

    // Contour controls
    const enabledCheckbox = container.querySelector<HTMLInputElement>('[data-contour="enabled"]');
    const strokeWidthSlider = container.querySelector<HTMLInputElement>('[data-contour="strokeWidth"]');
    const strokeColorInput = container.querySelector<HTMLInputElement>('[data-contour="strokeColor"]');
    const opacitySlider = container.querySelector<HTMLInputElement>('[data-contour="opacity"]');
    const thresholdsSelect = container.querySelector<HTMLSelectElement>('[data-contour="thresholds"]');

    enabledCheckbox?.addEventListener('change', (e) => {
      const target = e.target as HTMLInputElement;
      if (target.checked) {
        graph.enableContour?.();
      } else {
        graph.disableContour?.();
      }
    });

    strokeWidthSlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="strokeWidth"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setContourConfig?.({ strokeWidth: value });
    });

    strokeColorInput?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      graph.setContourConfig?.({ strokeColor: target.value });
    });

    opacitySlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="opacity"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setContourConfig?.({ opacity: value });
    });

    thresholdsSelect?.addEventListener('change', (e) => {
      const target = e.target as HTMLSelectElement;
      const thresholds = parseThresholds(target.value);
      graph.setContourConfig?.({ thresholds });
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
 * Clustered Graph with Contours - Shows cluster boundaries
 */
export const ClusteredContours: Story = {
  args: {
    thresholds: [0.3, 0.5, 0.7],
    strokeWidth: 2,
    strokeColor: '#ffffff',
    opacity: 0.9,
    enabled: true,
  },
  render: (args) => {
    const container = createContourContainer('Clustered Graph with Contour Lines', args);
    container.dataset.contourConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.contourConfig || '{}') as ContourConfig;
    const data = generateClusteredGraph(5, 120);
    await initContourGraph(container, data, config);
  },
};

/**
 * Topographic Style - Multiple threshold levels like a topographic map
 */
export const TopographicStyle: Story = {
  args: {
    thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    strokeWidth: 1,
    strokeColor: '#00ff88',
    opacity: 0.8,
    enabled: true,
  },
  render: (args) => {
    const container = createContourContainer('Topographic Style Contours', args);
    container.dataset.contourConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.contourConfig || '{}') as ContourConfig;
    const data = generateClusteredGraph(8, 100);
    await initContourGraph(container, data, config);
  },
};

/**
 * Spiral Galaxy with Contours
 */
export const SpiralContours: Story = {
  args: {
    thresholds: [0.3, 0.6],
    strokeWidth: 2.5,
    strokeColor: '#ff6644',
    opacity: 0.85,
    enabled: true,
  },
  render: (args) => {
    const container = createContourContainer('Spiral Galaxy with Contours', args);
    container.dataset.contourConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.contourConfig || '{}') as ContourConfig;
    const data = generateSpiralGraph(1500, 4);
    await initContourGraph(container, data, config);
  },
};

/**
 * Bold Single Contour - Highlight the main density boundary
 */
export const BoldSingleContour: Story = {
  args: {
    thresholds: [0.5],
    strokeWidth: 4,
    strokeColor: '#ffcc00',
    opacity: 1.0,
    enabled: true,
  },
  render: (args) => {
    const container = createContourContainer('Bold Single Contour Line', args);
    container.dataset.contourConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.contourConfig || '{}') as ContourConfig;
    const data = generateClusteredGraph(4, 150);
    await initContourGraph(container, data, config);
  },
};
