import type { Meta, StoryObj } from '@storybook/html';
import './heroine-graph.css';
import { generateRandomGraph, generateSocialNetwork, generateTree, type GraphData } from './graph-generators.ts';

interface BasicGraphArgs {
  nodeCount: number;
  showStats: boolean;
}

interface HeroineGraph {
  frameStats: { fps?: number; avgFrameTime?: number } | null;
  nodeCount: number;
  edgeCount: number;
  resize: (width: number, height: number) => void;
  load: (data: GraphData) => Promise<void>;
  startSimulation: () => void;
  pauseSimulation: () => void;
  fitToView: () => void;
  pan: (dx: number, dy: number) => void;
  zoom: (factor: number) => void;
}

interface HeroineGraphModule {
  createHeroineGraph: (options: { canvas: HTMLCanvasElement; debug?: boolean }) => Promise<HeroineGraph>;
  getSupportInfo: () => Promise<{ supported: boolean; reason?: string }>;
}

const meta: Meta<BasicGraphArgs> = {
  title: 'HeroineGraph/Basic',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'Basic graph visualization examples using HeroineGraph WebGPU renderer.',
      },
    },
  },
  argTypes: {
    nodeCount: {
      control: { type: 'select' },
      options: [100, 500, 1000, 5000, 10000],
      description: 'Number of nodes in the graph',
    },
    showStats: {
      control: 'boolean',
      description: 'Show FPS and frame statistics',
    },
  },
};

export default meta;
type Story = StoryObj<BasicGraphArgs>;

/**
 * Create the graph container HTML
 */
function createContainer(title: string, showStats = true): HTMLDivElement {
  const container = document.createElement('div');
  container.className = 'heroine-graph-container';
  container.innerHTML = `
    <div class="heroine-graph-header">
      <h3 class="heroine-graph-title">${title}</h3>
      ${showStats ? `
        <div class="heroine-graph-stats">
          <div class="heroine-graph-stat">
            <span class="heroine-graph-stat-label">FPS:</span>
            <span class="heroine-graph-stat-value" data-stat="fps">--</span>
          </div>
          <div class="heroine-graph-stat">
            <span class="heroine-graph-stat-label">Frame:</span>
            <span class="heroine-graph-stat-value" data-stat="frameTime">--</span>
          </div>
          <div class="heroine-graph-stat">
            <span class="heroine-graph-stat-label">Nodes:</span>
            <span class="heroine-graph-stat-value" data-stat="nodes">--</span>
          </div>
          <div class="heroine-graph-stat">
            <span class="heroine-graph-stat-label">Edges:</span>
            <span class="heroine-graph-stat-value" data-stat="edges">--</span>
          </div>
        </div>
      ` : ''}
    </div>
    <canvas data-graph-canvas></canvas>
    <div class="heroine-graph-loading">
      <div class="heroine-graph-spinner"></div>
      <div data-loading-text>Initializing HeroineGraph...</div>
    </div>
    <div class="heroine-graph-controls" style="display: none;">
      <button class="heroine-graph-btn secondary" data-action="toggle-sim">Pause</button>
      <button class="heroine-graph-btn secondary" data-action="fit-view">Fit View</button>
      <button class="heroine-graph-btn secondary" data-action="reset">Reset</button>
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
 * Initialize the graph (async)
 */
async function initGraph(container: HTMLElement, graphData: GraphData): Promise<HeroineGraph | undefined> {
  const canvas = container.querySelector<HTMLCanvasElement>('[data-graph-canvas]')!;
  const loading = container.querySelector<HTMLElement>('.heroine-graph-loading')!;
  const loadingText = container.querySelector<HTMLElement>('[data-loading-text]')!;
  const controls = container.querySelector<HTMLElement>('.heroine-graph-controls')!;

  try {
    // Wait for container to be laid out with valid dimensions
    console.log('[initGraph] Waiting for layout...');
    await waitForLayout(container);
    console.log('[initGraph] Layout ready');

    // Size canvas BEFORE creating graph context
    const rect = container.getBoundingClientRect();
    console.log('[initGraph] Container rect:', rect);
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    console.log('[initGraph] Canvas sized to:', canvas.width, 'x', canvas.height);

    // Dynamic import of the library
    loadingText.textContent = 'Loading HeroineGraph...';
    const { createHeroineGraph, getSupportInfo } = await import('../dist/heroine-graph.esm.js') as HeroineGraphModule;

    // Check support
    loadingText.textContent = 'Checking WebGPU support...';
    const support = await getSupportInfo();

    if (!support.supported) {
      throw new Error(support.reason || 'WebGPU is not supported in this browser');
    }

    // Create graph
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
    resizeCanvas();

    // Load data
    loadingText.textContent = 'Loading graph data...';
    await graph.load(graphData);

    // Hide loading, show controls
    loading.style.display = 'none';
    controls.style.display = 'flex';

    // Update stats
    const fpsEl = container.querySelector<HTMLElement>('[data-stat="fps"]');
    const frameTimeEl = container.querySelector<HTMLElement>('[data-stat="frameTime"]');
    const nodesEl = container.querySelector<HTMLElement>('[data-stat="nodes"]');
    const edgesEl = container.querySelector<HTMLElement>('[data-stat="edges"]');

    if (fpsEl) {
      setInterval(() => {
        const stats = graph.frameStats;
        if (stats) {
          fpsEl.textContent = stats.fps?.toFixed(0) || '--';
          if (frameTimeEl) {
            frameTimeEl.textContent = stats.avgFrameTime ? `${stats.avgFrameTime.toFixed(1)}ms` : '--';
          }
        }
        if (nodesEl) nodesEl.textContent = graph.nodeCount?.toLocaleString() || '--';
        if (edgesEl) edgesEl.textContent = graph.edgeCount?.toLocaleString() || '--';
      }, 100);
    }

    // Control buttons
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

    container.querySelector('[data-action="reset"]')?.addEventListener('click', () => {
      graph.load(graphData);
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
        const dx = e.clientX - lastMouse.x;
        const dy = e.clientY - lastMouse.y;
        graph.pan(dx, dy);
        lastMouse = { x: e.clientX, y: e.clientY };
      }
    });

    canvas.addEventListener('mouseup', () => { isDragging = false; });
    canvas.addEventListener('mouseleave', () => { isDragging = false; });

    canvas.addEventListener('wheel', (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      graph.zoom(factor);
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
 * Random Graph - Basic visualization with random connections
 */
export const RandomGraph: Story = {
  args: {
    nodeCount: 1000,
    showStats: true,
  },
  render: (args) => {
    const container = createContainer('Random Graph', args.showStats);
    container.dataset.nodeCount = String(args.nodeCount);
    return container;
  },
  play: async ({ canvasElement }) => {
    console.log('[Story] canvasElement:', canvasElement);
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    console.log('[Story] container:', container);
    if (!container) {
      console.error('[Story] Container not found!');
      return;
    }
    const nodeCount = parseInt(container.dataset.nodeCount || '1000', 10);
    const data = generateRandomGraph(nodeCount);
    await initGraph(container, data);
  },
};

/**
 * Social Network - Demonstrates different node sizes for influencers
 */
export const SocialNetwork: Story = {
  args: {
    nodeCount: 500,
    showStats: true,
  },
  render: (args) => {
    const container = createContainer('Social Network', args.showStats);
    container.dataset.nodeCount = String(args.nodeCount);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const nodeCount = parseInt(container.dataset.nodeCount || '500', 10);
    const data = generateSocialNetwork(nodeCount);
    await initGraph(container, data);
  },
};

/**
 * Tree Structure - Hierarchical layout
 */
export const TreeStructure: Story = {
  args: {
    showStats: true,
    nodeCount: 100, // Not used but required by type
  },
  render: (args) => {
    const container = createContainer('Tree Structure', args.showStats);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const data = generateTree(5, 3);
    await initGraph(container, data);
  },
};

/**
 * Large Graph - Stress test with 10K nodes
 */
export const LargeGraph: Story = {
  args: {
    nodeCount: 10000,
    showStats: true,
  },
  render: (args) => {
    const container = createContainer(`Large Graph (${args.nodeCount.toLocaleString()} nodes)`, args.showStats);
    container.dataset.nodeCount = String(args.nodeCount);
    return container;
  },
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (!container) return;
    const nodeCount = parseInt(container.dataset.nodeCount || '10000', 10);
    const data = generateRandomGraph(nodeCount);
    await initGraph(container, data);
  },
};
