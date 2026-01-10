import type { Meta, StoryObj } from '@storybook/html';
import './heroine-graph.css';
import { generateClusteredGraph, generateSpiralGraph, type GraphData } from './graph-generators.ts';

interface LabelConfig {
  fontSize: number;
  fontColor: string;
  minZoom: number;
  maxLabels: number;
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
  enableLabels?: (config?: Partial<LabelConfig>) => Promise<void>;
  disableLabels?: () => void;
  setLabelsConfig?: (config: Partial<LabelConfig>) => void;
  setLabels?: (labels: LabelData[]) => void;
}

interface LabelData {
  nodeId: number;
  text: string;
  x: number;
  y: number;
  priority: number;
  minZoom?: number;
}

interface HeroineGraphModule {
  createHeroineGraph: (options: { canvas: HTMLCanvasElement; debug?: boolean }) => Promise<HeroineGraph>;
  getSupportInfo: () => Promise<{ supported: boolean; reason?: string }>;
}

const meta: Meta<LabelConfig> = {
  title: 'HeroineGraph/Labels Layer',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'MSDF text labels for node identification with priority-based culling and collision detection.',
      },
    },
  },
  argTypes: {
    fontSize: {
      control: { type: 'range', min: 8, max: 32, step: 1 },
      description: 'Font size in pixels',
    },
    fontColor: {
      control: 'color',
      description: 'Text color',
    },
    minZoom: {
      control: { type: 'range', min: 0.1, max: 2.0, step: 0.1 },
      description: 'Minimum zoom level to show labels',
    },
    maxLabels: {
      control: { type: 'range', min: 10, max: 500, step: 10 },
      description: 'Maximum visible labels',
    },
    enabled: {
      control: 'boolean',
      description: 'Enable/disable labels layer',
    },
  },
};

export default meta;
type Story = StoryObj<LabelConfig>;

/**
 * Create the labels graph container with controls
 */
function createLabelsContainer(title: string, initialConfig: LabelConfig): HTMLDivElement {
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
    <div class="heroine-graph-labels-controls" style="display: none;">
      <h4>Labels Controls</h4>
      <div class="heroine-graph-control-row">
        <label>Enabled</label>
        <input type="checkbox" data-labels="enabled" ${initialConfig.enabled ? 'checked' : ''}>
      </div>
      <div class="heroine-graph-control-row">
        <label>Font Size: <span data-value="fontSize">${initialConfig.fontSize}</span>px</label>
        <input type="range" data-labels="fontSize" min="8" max="32" step="1" value="${initialConfig.fontSize}">
      </div>
      <div class="heroine-graph-control-row">
        <label>Min Zoom: <span data-value="minZoom">${initialConfig.minZoom}</span></label>
        <input type="range" data-labels="minZoom" min="0.1" max="2.0" step="0.1" value="${initialConfig.minZoom}">
      </div>
      <div class="heroine-graph-control-row">
        <label>Max Labels: <span data-value="maxLabels">${initialConfig.maxLabels}</span></label>
        <input type="range" data-labels="maxLabels" min="10" max="500" step="10" value="${initialConfig.maxLabels}">
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
 * Generate labels from graph data
 */
function generateLabelsFromGraph(data: GraphData): LabelData[] {
  const labels: LabelData[] = [];
  const nodeCount = data.nodes.length;

  for (let i = 0; i < nodeCount; i++) {
    const node = data.nodes[i];
    // Use node label if available, otherwise use node ID
    const text = node.label || `Node ${i}`;
    // Priority based on node connections (higher degree = higher priority)
    const nodeIdStr = String(i);
    const connections = data.edges.filter(e => e.source === nodeIdStr || e.target === nodeIdStr).length;
    const priority = Math.min(1.0, connections / 10);

    labels.push({
      nodeId: i,
      text,
      x: node.x ?? 0,
      y: node.y ?? 0,
      priority,
      minZoom: priority > 0.5 ? 0.2 : 0.5, // High priority labels visible at lower zoom
    });
  }

  return labels;
}

/**
 * Initialize the graph with labels layer
 */
async function initLabelsGraph(
  container: HTMLElement,
  graphData: GraphData,
  labelsConfig: LabelConfig
): Promise<HeroineGraph | undefined> {
  const canvas = container.querySelector<HTMLCanvasElement>('[data-graph-canvas]')!;
  const loading = container.querySelector<HTMLElement>('.heroine-graph-loading')!;
  const loadingText = container.querySelector<HTMLElement>('[data-loading-text]')!;
  const controls = container.querySelector<HTMLElement>('.heroine-graph-controls')!;
  const labelsControls = container.querySelector<HTMLElement>('.heroine-graph-labels-controls')!;

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

    // Generate labels from graph data
    const labels = generateLabelsFromGraph(graphData);

    // Enable labels layer
    if (graph.enableLabels) {
      loadingText.textContent = 'Loading font atlas...';
      await graph.enableLabels(labelsConfig);
      graph.setLabels?.(labels);
    }

    // Hide loading, show controls
    loading.style.display = 'none';
    controls.style.display = 'flex';
    labelsControls.style.display = 'block';

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

    // Labels controls
    const enabledCheckbox = container.querySelector<HTMLInputElement>('[data-labels="enabled"]');
    const fontSizeSlider = container.querySelector<HTMLInputElement>('[data-labels="fontSize"]');
    const minZoomSlider = container.querySelector<HTMLInputElement>('[data-labels="minZoom"]');
    const maxLabelsSlider = container.querySelector<HTMLInputElement>('[data-labels="maxLabels"]');

    enabledCheckbox?.addEventListener('change', async (e) => {
      const target = e.target as HTMLInputElement;
      if (target.checked) {
        await graph.enableLabels?.();
      } else {
        graph.disableLabels?.();
      }
    });

    fontSizeSlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseInt(target.value);
      const valueDisplay = container.querySelector('[data-value="fontSize"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setLabelsConfig?.({ fontSize: value });
    });

    minZoomSlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseFloat(target.value);
      const valueDisplay = container.querySelector('[data-value="minZoom"]');
      if (valueDisplay) valueDisplay.textContent = value.toFixed(1);
      graph.setLabelsConfig?.({ minZoom: value });
    });

    maxLabelsSlider?.addEventListener('input', (e) => {
      const target = e.target as HTMLInputElement;
      const value = parseInt(target.value);
      const valueDisplay = container.querySelector('[data-value="maxLabels"]');
      if (valueDisplay) valueDisplay.textContent = String(value);
      graph.setLabelsConfig?.({ maxLabels: value });
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
 * Clustered Graph with Labels - Shows node labels with collision detection
 */
export const ClusteredGraphWithLabels: Story = {
  args: {
    fontSize: 14,
    fontColor: '#1f2937',
    minZoom: 0.3,
    maxLabels: 100,
    enabled: true,
  },
  render: (args) => {
    const container = createLabelsContainer('Clustered Graph with MSDF Labels', args);
    container.dataset.labelsConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.labelsConfig || '{}') as LabelConfig;
    const data = generateClusteredGraph(5, 50);
    await initLabelsGraph(container, data, config);
  },
};

/**
 * Spiral Galaxy with Labels - Labels on spiral pattern
 */
export const SpiralGalaxyWithLabels: Story = {
  args: {
    fontSize: 12,
    fontColor: '#374151',
    minZoom: 0.5,
    maxLabels: 150,
    enabled: true,
  },
  render: (args) => {
    const container = createLabelsContainer('Spiral Galaxy with Labels', args);
    container.dataset.labelsConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.labelsConfig || '{}') as LabelConfig;
    const data = generateSpiralGraph(500, 3);
    await initLabelsGraph(container, data, config);
  },
};

/**
 * High Density Labels - Test collision detection with many labels
 */
export const HighDensityLabels: Story = {
  args: {
    fontSize: 10,
    fontColor: '#1f2937',
    minZoom: 0.2,
    maxLabels: 200,
    enabled: true,
  },
  render: (args) => {
    const container = createLabelsContainer('High Density Labels (1000 nodes)', args);
    container.dataset.labelsConfig = JSON.stringify(args);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const config = JSON.parse(container.dataset.labelsConfig || '{}') as LabelConfig;
    const data = generateClusteredGraph(8, 125);
    await initLabelsGraph(container, data, config);
  },
};
