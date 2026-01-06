import type { Meta, StoryObj } from '@storybook/html';
import './heroine-graph.css';
import type { GraphData } from './graph-generators.ts';

interface ContourStoryArgs {
  preset: string;
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
  enableContour?: (config?: Record<string, unknown>) => void;
  disableContour?: () => void;
  setContourConfig?: (config: Record<string, unknown>) => void;
  enableHeatmap?: (config?: Record<string, unknown>) => void;
  setHeatmapConfig?: (config: Record<string, unknown>) => void;
}

interface HeroineGraphModule {
  createHeroineGraph: (options: { canvas: HTMLCanvasElement; debug?: boolean }) => Promise<HeroineGraph>;
  getSupportInfo: () => Promise<{ supported: boolean; reason?: string }>;
}

const meta: Meta<ContourStoryArgs> = {
  title: 'HeroineGraph/Contour Layer',
  parameters: {
    layout: 'fullscreen',
  },
};

export default meta;
type Story = StoryObj<ContourStoryArgs>;

/**
 * Generate simple blob data - just dense circular clusters
 * Perfect for contour visualization
 */
function generateBlobData(blobCount: number, nodesPerBlob: number): GraphData {
  const nodes: GraphData['nodes'] = [];
  const edges: GraphData['edges'] = [];
  let nodeId = 0;

  // Place blobs in a grid pattern
  const cols = Math.ceil(Math.sqrt(blobCount));
  const spacing = 200;

  for (let b = 0; b < blobCount; b++) {
    const col = b % cols;
    const row = Math.floor(b / cols);
    const centerX = (col - (cols - 1) / 2) * spacing;
    const centerY = (row - Math.floor(blobCount / cols) / 2) * spacing;
    const hue = (b / blobCount) * 360;

    // Create a tight circular blob
    for (let i = 0; i < nodesPerBlob; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * 30; // Small radius = tight cluster

      nodes.push({
        id: `n${nodeId}`,
        x: centerX + Math.cos(angle) * r,
        y: centerY + Math.sin(angle) * r,
        radius: 3,
        color: `hsl(${hue}, 70%, 50%)`,
      });
      nodeId++;
    }
  }

  // Minimal edges - just connect a few within each blob
  for (let b = 0; b < blobCount; b++) {
    const start = b * nodesPerBlob;
    for (let i = 0; i < 5; i++) {
      const src = start + Math.floor(Math.random() * nodesPerBlob);
      const tgt = start + Math.floor(Math.random() * nodesPerBlob);
      if (src !== tgt) {
        edges.push({ source: `n${src}`, target: `n${tgt}`, color: '#333', width: 0.5 });
      }
    }
  }

  return { nodes, edges };
}

/**
 * Create container with inline real-time controls
 */
function createContainer(): HTMLDivElement {
  const container = document.createElement('div');
  container.className = 'heroine-graph-container';
  container.innerHTML = `
    <div class="heroine-graph-header">
      <h3 class="heroine-graph-title">Contour Layer</h3>
      <div class="heroine-graph-stats">
        <span class="heroine-graph-stat-label">FPS:</span>
        <span class="heroine-graph-stat-value" data-stat="fps">--</span>
      </div>
    </div>
    <canvas data-graph-canvas></canvas>
    <div class="heroine-graph-loading">
      <div class="heroine-graph-spinner"></div>
      <div data-loading-text>Initializing...</div>
    </div>

    <!-- Real-time controls panel -->
    <div class="contour-controls" style="display:none; position:absolute; top:60px; left:10px; background:rgba(0,0,0,0.85); padding:15px; border-radius:8px; color:white; font-size:12px; width:220px; z-index:100;">
      <div style="margin-bottom:12px; font-weight:bold; border-bottom:1px solid #444; padding-bottom:8px;">Contour Settings</div>

      <label style="display:block; margin-bottom:8px;">
        Threshold: <span data-value="threshold">0.5</span>
        <input type="range" data-control="threshold" min="0.1" max="0.9" step="0.05" value="0.5" style="width:100%;">
      </label>

      <label style="display:block; margin-bottom:8px;">
        Line Width: <span data-value="lineWidth">3</span>
        <input type="range" data-control="lineWidth" min="1" max="10" step="0.5" value="3" style="width:100%;">
      </label>

      <label style="display:block; margin-bottom:8px;">
        Line Color:
        <input type="color" data-control="strokeColor" value="#ff0000" style="width:100%; height:25px;">
      </label>

      <div style="margin:15px 0 12px; font-weight:bold; border-bottom:1px solid #444; padding-bottom:8px;">Heatmap Settings</div>

      <label style="display:block; margin-bottom:8px;">
        Radius: <span data-value="radius">40</span>
        <input type="range" data-control="radius" min="10" max="100" step="5" value="40" style="width:100%;">
      </label>

      <label style="display:block; margin-bottom:8px;">
        Max Density: <span data-value="maxDensity">10</span>
        <input type="range" data-control="maxDensity" min="1" max="50" step="1" value="10" style="width:100%;">
      </label>

      <label style="display:block; margin-bottom:8px;">
        Heatmap Opacity: <span data-value="heatmapOpacity">0.3</span>
        <input type="range" data-control="heatmapOpacity" min="0" max="1" step="0.1" value="0.3" style="width:100%;">
      </label>

      <div style="margin:15px 0 12px; font-weight:bold; border-bottom:1px solid #444; padding-bottom:8px;">Data</div>

      <label style="display:block; margin-bottom:8px;">
        Blobs: <span data-value="blobs">3</span>
        <input type="range" data-control="blobs" min="1" max="6" step="1" value="3" style="width:100%;">
      </label>

      <label style="display:block; margin-bottom:8px;">
        Nodes/Blob: <span data-value="nodesPerBlob">100</span>
        <input type="range" data-control="nodesPerBlob" min="20" max="300" step="20" value="100" style="width:100%;">
      </label>

      <button data-action="regenerate" style="width:100%; padding:8px; margin-top:10px; cursor:pointer;">Regenerate Data</button>
    </div>

    <div class="heroine-graph-controls" style="display: none;">
      <button class="heroine-graph-btn secondary" data-action="toggle-sim">Pause</button>
      <button class="heroine-graph-btn secondary" data-action="fit-view">Fit View</button>
    </div>
  `;
  return container;
}

async function waitForLayout(container: HTMLElement): Promise<void> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      if (container.isConnected && container.getBoundingClientRect().width > 0) {
        resolve();
      } else if (Date.now() - start > 5000) {
        reject(new Error('Timeout'));
      } else {
        requestAnimationFrame(check);
      }
    };
    requestAnimationFrame(check);
  });
}

async function initGraph(container: HTMLElement): Promise<void> {
  const canvas = container.querySelector<HTMLCanvasElement>('[data-graph-canvas]')!;
  const loading = container.querySelector<HTMLElement>('.heroine-graph-loading')!;
  const loadingText = container.querySelector<HTMLElement>('[data-loading-text]')!;
  const controls = container.querySelector<HTMLElement>('.heroine-graph-controls')!;
  const contourControls = container.querySelector<HTMLElement>('.contour-controls')!;

  try {
    await waitForLayout(container);

    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;

    loadingText.textContent = 'Loading...';
    const { createHeroineGraph, getSupportInfo } = await import('../dist/heroine-graph.esm.js') as HeroineGraphModule;

    const support = await getSupportInfo();
    if (!support.supported) {
      throw new Error(support.reason || 'WebGPU not supported');
    }

    loadingText.textContent = 'Initializing WebGPU...';
    const graph = await createHeroineGraph({ canvas, debug: true });

    // Handle resize
    window.addEventListener('resize', () => {
      const r = container.getBoundingClientRect();
      canvas.width = r.width * window.devicePixelRatio;
      canvas.height = r.height * window.devicePixelRatio;
      graph.resize(canvas.width, canvas.height);
    });

    // State
    let currentBlobs = 3;
    let currentNodesPerBlob = 100;

    // Load initial data
    const loadData = async () => {
      loadingText.textContent = 'Generating blobs...';
      loading.style.display = 'flex';
      const data = generateBlobData(currentBlobs, currentNodesPerBlob);
      await graph.load(data);
      loading.style.display = 'none';
    };

    await loadData();

    // Enable heatmap
    graph.enableHeatmap?.({
      opacity: 0.3,
      radius: 40,
      intensity: 0.3,
      maxDensity: 10,
    });

    // Enable contours with multiple thresholds at low values
    // (density values are typically 0.01-0.3 range)
    graph.enableContour?.({
      thresholds: [0.05, 0.1, 0.15, 0.2, 0.25],
      strokeWidth: 2,
      strokeColor: '#ff0000',
      opacity: 1.0,
    });

    loading.style.display = 'none';
    controls.style.display = 'flex';
    contourControls.style.display = 'block';

    // FPS display
    setInterval(() => {
      const fps = container.querySelector('[data-stat="fps"]');
      if (fps && graph.frameStats) {
        fps.textContent = graph.frameStats.fps?.toFixed(0) || '--';
      }
    }, 200);

    // Wire up real-time controls
    const wireControl = (name: string, handler: (value: number | string) => void) => {
      const input = container.querySelector<HTMLInputElement>(`[data-control="${name}"]`);
      const display = container.querySelector<HTMLElement>(`[data-value="${name}"]`);
      if (input) {
        input.addEventListener('input', () => {
          const val = input.type === 'color' ? input.value : parseFloat(input.value);
          if (display) display.textContent = String(val);
          handler(val);
        });
      }
    };

    // Contour controls
    wireControl('threshold', (val) => {
      graph.setContourConfig?.({ thresholds: [val as number] });
    });

    wireControl('lineWidth', (val) => {
      graph.setContourConfig?.({ strokeWidth: val as number });
    });

    wireControl('strokeColor', (val) => {
      graph.setContourConfig?.({ strokeColor: val as string });
    });

    // Heatmap controls
    wireControl('radius', (val) => {
      graph.setHeatmapConfig?.({ radius: val as number });
    });

    wireControl('maxDensity', (val) => {
      graph.setHeatmapConfig?.({ maxDensity: val as number });
    });

    wireControl('heatmapOpacity', (val) => {
      graph.setHeatmapConfig?.({ opacity: val as number });
    });

    // Data controls
    wireControl('blobs', (val) => {
      currentBlobs = val as number;
    });

    wireControl('nodesPerBlob', (val) => {
      currentNodesPerBlob = val as number;
    });

    container.querySelector('[data-action="regenerate"]')?.addEventListener('click', loadData);

    // Simulation toggle
    let paused = false;
    container.querySelector('[data-action="toggle-sim"]')?.addEventListener('click', (e) => {
      if (paused) {
        graph.startSimulation();
        (e.target as HTMLButtonElement).textContent = 'Pause';
      } else {
        graph.pauseSimulation();
        (e.target as HTMLButtonElement).textContent = 'Resume';
      }
      paused = !paused;
    });

    container.querySelector('[data-action="fit-view"]')?.addEventListener('click', () => {
      graph.fitToView();
    });

    // Pan/zoom
    let dragging = false;
    let lastMouse = { x: 0, y: 0 };

    canvas.addEventListener('mousedown', (e) => {
      dragging = true;
      lastMouse = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mousemove', (e) => {
      if (dragging) {
        graph.pan(e.clientX - lastMouse.x, e.clientY - lastMouse.y);
        lastMouse = { x: e.clientX, y: e.clientY };
      }
    });

    canvas.addEventListener('mouseup', () => { dragging = false; });
    canvas.addEventListener('mouseleave', () => { dragging = false; });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      graph.zoom(e.deltaY > 0 ? 0.9 : 1.1);
    }, { passive: false });

  } catch (err) {
    loading.innerHTML = `<div class="heroine-graph-error"><h3>Error</h3><p>${(err as Error).message}</p></div>`;
    console.error(err);
  }
}

export const Interactive: Story = {
  render: () => createContainer(),
  play: async ({ canvasElement }) => {
    const container = canvasElement.querySelector<HTMLElement>('.heroine-graph-container');
    if (container) await initGraph(container);
  },
};
