import type { Meta, StoryObj } from '@storybook/html';
import './graphmother.css';
import { generateClusteredGraph, generateSpiralGraph, type GraphData } from './graph-generators.ts';
import { registerStoryCleanup, runStoryCleanups } from './story-cleanup.ts';

interface MetaballConfig {
  threshold: number;
  blendRadius: number;
  nodeRadius: number;
  fillColor: string;
  opacity: number;
  outlineOnly: boolean;
  outlineWidth: number;
  enabled: boolean;
}

interface GraphMother {
  frameStats: { fps?: number; avgFrameTime?: number } | null;
  nodeCount: number;
  resize: (width: number, height: number) => void;
  load: (data: GraphData) => Promise<void>;
  startSimulation: () => void;
  pauseSimulation: () => void;
  fitToView: () => void;
  pan: (dx: number, dy: number) => void;
  zoom: (factor: number) => void;
  enableMetaball?: (config?: Partial<MetaballConfig>) => void;
  disableMetaball?: () => void;
  setMetaballConfig?: (config: Partial<MetaballConfig>) => void;
  dispose: () => void;
}

interface GraphMotherModule {
  createGraphMother: (options: { canvas: HTMLCanvasElement; debug?: boolean }) => Promise<GraphMother>;
  getSupportInfo: () => Promise<{ supported: boolean; reason?: string }>;
}

const meta: Meta<MetaballConfig> = {
  title: 'GraphMother/Metaball Layer',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'Metaball visualization layer rendering smooth blob-like shapes around node clusters using screen-space SDF evaluation with quadratic smooth minimum for organic blending.',
      },
    },
  },
  argTypes: {
    threshold: {
      control: { type: 'range', min: 0.1, max: 2.0, step: 0.1 },
      description: 'SDF threshold for surface boundary',
    },
    blendRadius: {
      control: { type: 'range', min: 10, max: 200, step: 10 },
      description: 'Smooth minimum blend radius',
    },
    nodeRadius: {
      control: { type: 'range', min: 5, max: 100, step: 5 },
      description: 'Effective radius of each node',
    },
    fillColor: {
      control: 'color',
      description: 'Metaball fill color',
    },
    opacity: {
      control: { type: 'range', min: 0.1, max: 1.0, step: 0.1 },
      description: 'Layer opacity',
    },
    outlineOnly: {
      control: 'boolean',
      description: 'Render only the outline',
    },
    outlineWidth: {
      control: { type: 'range', min: 1, max: 10, step: 1 },
      description: 'Outline width (when outlineOnly is true)',
    },
    enabled: {
      control: 'boolean',
      description: 'Enable/disable metaball layer',
    },
  },
};

export default meta;
type Story = StoryObj<MetaballConfig>;

/**
 * Create the metaball graph container with controls
 */
function createMetaballContainer(title: string, initialConfig: MetaballConfig): HTMLDivElement {
  const container = document.createElement('div');
  container.className = 'graphmother-container';
  container.innerHTML = `
    <div class="graphmother-header">
      <h3 class="graphmother-title">${title}</h3>
      <div class="graphmother-stats">
        <div class="graphmother-stat">
          <span class="graphmother-stat-label">FPS:</span>
          <span class="graphmother-stat-value" data-stat="fps">--</span>
        </div>
        <div class="graphmother-stat">
          <span class="graphmother-stat-label">Nodes:</span>
          <span class="graphmother-stat-value" data-stat="nodes">--</span>
        </div>
      </div>
    </div>
    <canvas data-graph-canvas></canvas>
    <div class="graphmother-loading">
      <div class="graphmother-spinner"></div>
      <div data-loading-text>Initializing GraphMother...</div>
    </div>
    <div class="graphmother-metaball-controls" style="display: none;">
      <h4>Metaball Controls</h4>
      <div class="graphmother-control-row">
        <label>Enabled</label>
        <input type="checkbox" data-metaball="enabled" ${initialConfig.enabled ? 'checked' : ''}>
      </div>
      <div class="graphmother-control-row">
        <label>Threshold: <span data-value="threshold">${initialConfig.threshold}</span></label>
        <input type="range" data-metaball="threshold" min="0.1" max="2.0" step="0.1" value="${initialConfig.threshold}">
      </div>
      <div class="graphmother-control-row">
        <label>Blend Radius: <span data-value="blendRadius">${initialConfig.blendRadius}</span></label>
        <input type="range" data-metaball="blendRadius" min="10" max="200" step="10" value="${initialConfig.blendRadius}">
      </div>
      <div class="graphmother-control-row">
        <label>Node Radius: <span data-value="nodeRadius">${initialConfig.nodeRadius}</span></label>
        <input type="range" data-metaball="nodeRadius" min="5" max="100" step="5" value="${initialConfig.nodeRadius}">
      </div>
      <div class="graphmother-control-row">
        <label>Fill Color</label>
        <input type="color" data-metaball="fillColor" value="${initialConfig.fillColor}">
      </div>
      <div class="graphmother-control-row">
        <label>Opacity: <span data-value="opacity">${initialConfig.opacity}</span></label>
        <input type="range" data-metaball="opacity" min="0.1" max="1.0" step="0.1" value="${initialConfig.opacity}">
      </div>
      <div class="graphmother-control-row">
        <label>Outline Only</label>
        <input type="checkbox" data-metaball="outlineOnly" ${initialConfig.outlineOnly ? 'checked' : ''}>
      </div>
      <div class="graphmother-control-row">
        <label>Outline Width: <span data-value="outlineWidth">${initialConfig.outlineWidth}</span></label>
        <input type="range" data-metaball="outlineWidth" min="1" max="10" step="1" value="${initialConfig.outlineWidth}">
      </div>
    </div>
    <div class="graphmother-controls" style="display: none;">
      <button class="graphmother-btn secondary" data-action="toggle-sim">Pause</button>
      <button class="graphmother-btn secondary" data-action="fit-view">Fit View</button>
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
 * Initialize the graph with metaball layer
 */
async function initMetaballGraph(
  container: HTMLElement,
  graphData: GraphData,
  metaballConfig: MetaballConfig
): Promise<GraphMother | undefined> {
  const canvas = container.querySelector<HTMLCanvasElement>('[data-graph-canvas]')!;
  const loading = container.querySelector<HTMLElement>('.graphmother-loading')!;
  const loadingText = container.querySelector<HTMLElement>('[data-loading-text]')!;
  const controls = container.querySelector<HTMLElement>('.graphmother-controls')!;
  const metaballControls = container.querySelector<HTMLElement>('.graphmother-metaball-controls')!;

  try {
    // Tear down graphs/timers/listeners from previously rendered stories
    runStoryCleanups();

    await waitForLayout(container);

    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;

    loadingText.textContent = 'Loading GraphMother...';
    const { createGraphMother, getSupportInfo } = await import('../dist/graphmother.esm.js') as GraphMotherModule;

    loadingText.textContent = 'Checking WebGPU support...';
    const support = await getSupportInfo();

    if (!support.supported) {
      throw new Error(support.reason || 'WebGPU is not supported in this browser');
    }

    loadingText.textContent = 'Initializing WebGPU...';
    const graph = await createGraphMother({ canvas, debug: false });

    const resizeCanvas = () => {
      const r = container.getBoundingClientRect();
      canvas.width = r.width * window.devicePixelRatio;
      canvas.height = r.height * window.devicePixelRatio;
      graph.resize(canvas.width, canvas.height);
    };
    window.addEventListener('resize', resizeCanvas);

    let statsInterval: ReturnType<typeof setInterval> | undefined;
    registerStoryCleanup(() => {
      window.removeEventListener('resize', resizeCanvas);
      if (statsInterval !== undefined) clearInterval(statsInterval);
      graph.dispose();
    });

    loadingText.textContent = 'Loading graph data...';
    await graph.load(graphData);

    // Enable metaball layer
    if (graph.enableMetaball) {
      graph.enableMetaball(metaballConfig);
    }

    loading.style.display = 'none';
    controls.style.display = 'flex';
    metaballControls.style.display = 'block';

    // Update stats
    const fpsEl = container.querySelector<HTMLElement>('[data-stat="fps"]');
    const nodesEl = container.querySelector<HTMLElement>('[data-stat="nodes"]');

    if (fpsEl) {
      statsInterval = setInterval(() => {
        const stats = graph.frameStats;
        if (stats) {
          fpsEl.textContent = stats.fps?.toFixed(0) || '--';
        }
        if (nodesEl) nodesEl.textContent = graph.nodeCount?.toLocaleString() || '--';
      }, 100);
    }

    // Metaball controls
    const enabledCheckbox = container.querySelector<HTMLInputElement>('[data-metaball="enabled"]');
    const thresholdSlider = container.querySelector<HTMLInputElement>('[data-metaball="threshold"]');
    const blendRadiusSlider = container.querySelector<HTMLInputElement>('[data-metaball="blendRadius"]');
    const nodeRadiusSlider = container.querySelector<HTMLInputElement>('[data-metaball="nodeRadius"]');
    const fillColorInput = container.querySelector<HTMLInputElement>('[data-metaball="fillColor"]');
    const opacitySlider = container.querySelector<HTMLInputElement>('[data-metaball="opacity"]');
    const outlineOnlyCheckbox = container.querySelector<HTMLInputElement>('[data-metaball="outlineOnly"]');
    const outlineWidthSlider = container.querySelector<HTMLInputElement>('[data-metaball="outlineWidth"]');

    enabledCheckbox?.addEventListener('change', (e) => {
      const target = e.target as HTMLInputElement;
      if (target.checked) {
        graph.enableMetaball?.();
      } else {
        graph.disableMetaball?.();
      }
    });

    thresholdSlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="threshold"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setMetaballConfig?.({ threshold: value });
    });

    blendRadiusSlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="blendRadius"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setMetaballConfig?.({ blendRadius: value });
    });

    nodeRadiusSlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="nodeRadius"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setMetaballConfig?.({ nodeRadius: value });
    });

    fillColorInput?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      graph.setMetaballConfig?.({ fillColor: target.value });
    });

    opacitySlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="opacity"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setMetaballConfig?.({ opacity: value });
    });

    outlineOnlyCheckbox?.addEventListener('change', (e) => {
      const target = e.target as HTMLInputElement;
      graph.setMetaballConfig?.({ outlineOnly: target.checked });
    });

    outlineWidthSlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="outlineWidth"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setMetaballConfig?.({ outlineWidth: value });
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
      <div class="graphmother-error">
        <h3>Initialization Error</h3>
        <p>${error.message}</p>
        <p style="margin-top: 12px; font-size: 12px;">
          WebGPU requires Chrome 113+, Edge 113+, or Safari 18+
        </p>
      </div>
    `;
    console.error('GraphMother initialization failed:', err);
  }
}

/**
 * Clustered Graph with Metaballs - Organic cluster boundaries
 */
export const OrganicClusters: Story = {
  args: {
    threshold: 0.8,
    blendRadius: 80,
    nodeRadius: 30,
    fillColor: '#4488ff',
    opacity: 0.6,
    outlineOnly: false,
    outlineWidth: 3,
    enabled: true,
  },
  render: (args) => {
    const container = createMetaballContainer('Organic Cluster Boundaries', args);
    container.dataset.metaballConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.graphmother-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.metaballConfig || '{}') as MetaballConfig;
    const data = generateClusteredGraph(5, 80);
    await initMetaballGraph(container, data, config);
  },
};

/**
 * Outline Mode - Just the boundary outline
 */
export const OutlineMode: Story = {
  args: {
    threshold: 0.7,
    blendRadius: 60,
    nodeRadius: 25,
    fillColor: '#ff6644',
    opacity: 0.9,
    outlineOnly: true,
    outlineWidth: 4,
    enabled: true,
  },
  render: (args) => {
    const container = createMetaballContainer('Metaball Outlines Only', args);
    container.dataset.metaballConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.graphmother-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.metaballConfig || '{}') as MetaballConfig;
    const data = generateClusteredGraph(6, 100);
    await initMetaballGraph(container, data, config);
  },
};

/**
 * Tight Blobs - Smaller blend radius for distinct clusters
 */
export const TightBlobs: Story = {
  args: {
    threshold: 1.0,
    blendRadius: 30,
    nodeRadius: 20,
    fillColor: '#44ff88',
    opacity: 0.5,
    outlineOnly: false,
    outlineWidth: 2,
    enabled: true,
  },
  render: (args) => {
    const container = createMetaballContainer('Tight Cluster Blobs', args);
    container.dataset.metaballConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.graphmother-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.metaballConfig || '{}') as MetaballConfig;
    const data = generateClusteredGraph(8, 60);
    await initMetaballGraph(container, data, config);
  },
};

/**
 * Spiral Galaxy with Metaballs
 */
export const SpiralMetaballs: Story = {
  args: {
    threshold: 0.6,
    blendRadius: 100,
    nodeRadius: 35,
    fillColor: '#aa44ff',
    opacity: 0.4,
    outlineOnly: false,
    outlineWidth: 3,
    enabled: true,
  },
  render: (args) => {
    const container = createMetaballContainer('Spiral Galaxy Metaballs', args);
    container.dataset.metaballConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.graphmother-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.metaballConfig || '{}') as MetaballConfig;
    const data = generateSpiralGraph(1200, 3);
    await initMetaballGraph(container, data, config);
  },
};

/**
 * Large Blend - Very smooth merging between clusters
 */
export const LargeBlend: Story = {
  args: {
    threshold: 0.5,
    blendRadius: 150,
    nodeRadius: 40,
    fillColor: '#ffaa44',
    opacity: 0.5,
    outlineOnly: false,
    outlineWidth: 2,
    enabled: true,
  },
  render: (args) => {
    const container = createMetaballContainer('Large Blend Radius', args);
    container.dataset.metaballConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.graphmother-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.metaballConfig || '{}') as MetaballConfig;
    const data = generateClusteredGraph(4, 120);
    await initMetaballGraph(container, data, config);
  },
};
