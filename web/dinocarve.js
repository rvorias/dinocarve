/*
 * dinocarve.js: browser runtime for offline-compiled seam carving. One ES module, no dependencies.
 *
 * Data contract (one directory per image, written by the dinocarve compiler):
 *   carve.json  {"version":1, "width":W, "height":H, "max_remove":K, "never":65535,
 *                "image":"image.jpg", "rank":"rank.png"}
 *   image.jpg   the W x H source.
 *   rank.png    W x H, rank(x,y) = R*256 + G. After removing k seams, pixel (x,y) is kept iff
 *               rank >= k. Every row keeps exactly W-k pixels for k in [0,K], so the carved image
 *               is each row's kept pixels compacted left to right.
 *
 * <carved-image src="examples/gnome/">
 *   Fills its CSS box. Picks k so the carved aspect matches the box:
 *   k = clamp(W - round(H * boxWidth / boxHeight), 0, K). Outside that range (box narrower than
 *   W-K allows, or wider than W) the canvas is letterboxed with object-fit: contain, never
 *   cropped or stretched. With no CSS height the box takes the carved image's own aspect.
 *   Attributes: src (required, the directory), width-percent (optional, fixes the carved width
 *   at that percent of W and ignores the box aspect).
 *   Properties (read-only): k, width (carved width in source pixels), naturalWidth (W),
 *   maxRemove (K). The last two are 0 until the src has loaded.
 *   Event: "carved" with detail {k, width, ms, backend} after each render. Also sets data-backend.
 *   Rendering is lazy: nothing is fetched or drawn until the element is within 200px of the
 *   viewport. Resizes and attribute changes coalesce into one render per animation frame.
 *
 * <carved-gallery>
 *   A CSS grid of <carved-image> children with fixed-height rows, so each cell's aspect follows
 *   its column/row span and the viewport width. Size cells with grid-column / grid-row spans or
 *   aspect-ratio. Override the grid with --carved-gallery-min (column min width), --carved-gallery-row
 *   (row height) and --carved-gallery-gap.
 *
 * Backend: WebGPU when navigator.gpu yields an adapter and device (one shared GPUDevice; per
 * asset the pixels and ranks are uploaded once as storage buffers; a compute pass compacts each
 * row, a render pass draws the result). Otherwise a CPU loop with putImageData. The pixel output
 * of both is identical. If a WebGPU draw ever fails (lost device, failed submit), WebGPU is disabled
 * for the page and every element falls back to the CPU. Canvas resolution is (W-k) x H; CSS scales it.
 */

const assets = new Map();

function loadAsset(dir) {
  if (!assets.has(dir)) {
    const promise = (async () => {
      const meta = await (await fetch(new URL('carve.json', dir))).json();
      const [color, rank] = await Promise.all([meta.image, meta.rank].map(name => decode(new URL(name, dir))));
      const W = meta.width, H = meta.height, N = W * H, ranks = new Uint32Array(N);
      for (let i = 0; i < N; i++) ranks[i] = rank[i * 4] << 8 | rank[i * 4 + 1];
      return {W, H, K: meta.max_remove, color: new Uint32Array(color.buffer), rank: ranks, gpu: null};
    })();
    promise.catch(() => assets.delete(dir));
    assets.set(dir, promise);
  }
  return assets.get(dir);
}

// Without these options the browser premultiplies or color-converts, which corrupts rank bytes.
async function decode(url) {
  const blob = await (await fetch(url)).blob();
  const bmp = await createImageBitmap(blob, {premultiplyAlpha: 'none', colorSpaceConversion: 'none'});
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const g = canvas.getContext('2d', {willReadFrequently: true});
  g.drawImage(bmp, 0, 0);
  return g.getImageData(0, 0, bmp.width, bmp.height).data;
}

const COMPACT = /* wgsl */ `
struct P { w: u32, h: u32, k: u32, ow: u32 }
@group(0) @binding(0) var<storage, read> color: array<u32>;
@group(0) @binding(1) var<storage, read> rank: array<u32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let y = id.x;
  if (y >= p.h) { return; }
  let row = y * p.w;
  var j = y * p.ow;
  for (var x = 0u; x < p.w; x++) {
    if (rank[row + x] >= p.k) { outp[j] = color[row + x]; j++; }
  }
}`;

const DRAW = /* wgsl */ `
struct P { w: u32, h: u32, k: u32, ow: u32 }
@group(0) @binding(0) var<storage, read> px: array<u32>;
@group(0) @binding(1) var<uniform> p: P;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return unpack4x8unorm(px[u32(pos.y) * p.ow + u32(pos.x)]);
}`;

let gpuPromise, gpuBroken = false;

function gpu() {
  if (gpuBroken) return Promise.resolve(null);
  gpuPromise ??= (async () => {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const format = navigator.gpu.getPreferredCanvasFormat();
    const compact = device.createComputePipeline({
      layout: 'auto', compute: {module: device.createShaderModule({code: COMPACT}), entryPoint: 'main'},
    });
    const module = device.createShaderModule({code: DRAW});
    const draw = device.createRenderPipeline({
      layout: 'auto', vertex: {module, entryPoint: 'vs'}, fragment: {module, entryPoint: 'fs', targets: [{format}]},
    });
    device.lost.then(() => { gpuPromise = undefined; });
    return {device, format, compact, draw};
  })().catch(() => null);
  return gpuPromise;
}

function storage(device, data) {
  const buffer = device.createBuffer({size: data.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true});
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

// One renderer per element; the asset's input buffers are shared by every element using it.
function gpuRenderer(g, canvas, asset) {
  const {device} = g, {W, H} = asset;
  asset.gpu ??= {color: storage(device, asset.color), rank: storage(device, asset.rank)};
  const out = device.createBuffer({size: W * H * 4, usage: GPUBufferUsage.STORAGE});
  const params = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  const bind = (pipeline, buffers) => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}})),
  });
  const compactGroup = bind(g.compact, [asset.gpu.color, asset.gpu.rank, out, params]);
  const drawGroup = bind(g.draw, [out, params]);
  const context = canvas.getContext('webgpu');
  context.configure({device, format: g.format, alphaMode: 'opaque'});
  return {
    backend: 'webgpu',
    async draw(k) {
      device.queue.writeBuffer(params, 0, new Uint32Array([W, H, k, W - k]));
      const encoder = device.createCommandEncoder();
      const cpass = encoder.beginComputePass();
      cpass.setPipeline(g.compact);
      cpass.setBindGroup(0, compactGroup);
      cpass.dispatchWorkgroups(Math.ceil(H / 64));
      cpass.end();
      const rpass = encoder.beginRenderPass({colorAttachments: [{
        view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store',
      }]});
      rpass.setPipeline(g.draw);
      rpass.setBindGroup(0, drawGroup);
      rpass.draw(3);
      rpass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    },
  };
}

function cpuRenderer(canvas, asset) {
  const {W, H, color, rank} = asset, g = canvas.getContext('2d');
  return {
    backend: 'cpu',
    draw(k) {
      const out = new Uint32Array((W - k) * H);
      for (let i = 0, j = 0; i < W * H; i++) if (rank[i] >= k) out[j++] = color[i];
      g.putImageData(new ImageData(new Uint8ClampedArray(out.buffer), W - k, H), 0, 0);
    },
  };
}

const nearViewport = new IntersectionObserver(entries => {
  for (const e of entries) {
    e.target.visible = e.isIntersecting;
    if (e.isIntersecting) e.target.schedule();
  }
}, {rootMargin: '200px'});

const resized = new ResizeObserver(entries => { for (const e of entries) e.target.schedule(); });

const sheet = new CSSStyleSheet();
sheet.replaceSync(`
  :host { display: block; position: relative; }
  canvas { display: block; width: 100%; height: 100%; object-fit: contain; }
`);

class CarvedImage extends HTMLElement {
  static observedAttributes = ['src', 'width-percent'];
  #canvas;
  #state = null; // {dir, asset, renderer, k} once the src has loaded
  #loading = null;
  #frame = 0;
  visible = false;

  get k() { return this.#state?.k ?? 0; }
  get width() { return this.#state ? this.#state.asset.W - this.k : 0; }
  get naturalWidth() { return this.#state?.asset.W ?? 0; }
  get maxRemove() { return this.#state?.asset.K ?? 0; }

  connectedCallback() {
    if (!this.#canvas) {
      const root = this.attachShadow({mode: 'open'});
      root.adoptedStyleSheets = [sheet];
      this.#canvas = root.appendChild(document.createElement('canvas'));
    }
    nearViewport.observe(this);
    resized.observe(this);
  }

  disconnectedCallback() {
    nearViewport.unobserve(this);
    resized.unobserve(this);
    cancelAnimationFrame(this.#frame);
    this.#frame = 0;
  }

  attributeChangedCallback(name) {
    if (name === 'src' && this.#state) {
      // A canvas keeps its first context type, so a new src gets a fresh canvas.
      this.#state = null;
      this.#canvas.replaceWith(this.#canvas = document.createElement('canvas'));
    }
    this.schedule();
  }

  schedule() {
    if (!this.#frame && this.isConnected) this.#frame = requestAnimationFrame(() => { this.#frame = 0; this.#render(); });
  }

  #dir() {
    return new URL(this.getAttribute('src').replace(/\/?$/, '/'), document.baseURI).href;
  }

  #targetK({W, H, K}) {
    const pct = parseFloat(this.getAttribute('width-percent'));
    let target;
    if (Number.isFinite(pct)) target = Math.round(W * pct / 100);
    else {
      const box = this.getBoundingClientRect();
      if (!box.width || !box.height) return 0;
      target = Math.round(H * box.width / box.height);
    }
    return Math.min(Math.max(W - target, 0), K);
  }

  async #render() {
    if (!this.visible || !this.hasAttribute('src')) return;
    const dir = this.#dir();
    if (this.#state?.dir !== dir) {
      if (this.#loading !== dir) {
        this.#loading = dir;
        const [asset, g] = await Promise.all([loadAsset(dir), gpu()]);
        this.#loading = null;
        if (dir !== this.#dir()) return this.schedule();
        const renderer = g ? gpuRenderer(g, this.#canvas, asset) : cpuRenderer(this.#canvas, asset);
        this.#state = {dir, asset, renderer, k: -1};
        this.dataset.backend = renderer.backend;
        this.schedule();
      }
      return;
    }
    const state = this.#state, k = this.#targetK(state.asset);
    if (k === state.k) return;
    state.k = k;
    this.#canvas.width = state.asset.W - k;
    this.#canvas.height = state.asset.H;
    const t0 = performance.now();
    try {
      await state.renderer.draw(k);
    } catch (err) {
      if (state.renderer.backend !== 'webgpu') throw err;
      gpuBroken = true;
      this.#canvas.replaceWith(this.#canvas = document.createElement('canvas'));
      this.#canvas.width = state.asset.W - k;
      this.#canvas.height = state.asset.H;
      state.renderer = cpuRenderer(this.#canvas, state.asset);
      this.dataset.backend = 'cpu';
      state.renderer.draw(k);
    }
    const ms = performance.now() - t0;
    this.dispatchEvent(new CustomEvent('carved', {detail: {k, width: state.asset.W - k, ms, backend: state.renderer.backend}}));
  }
}

const gallerySheet = new CSSStyleSheet();
gallerySheet.replaceSync(`
  :where(carved-gallery) {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--carved-gallery-min, 140px)), 1fr));
    grid-auto-rows: var(--carved-gallery-row, 180px);
    grid-auto-flow: dense;
    gap: var(--carved-gallery-gap, 8px);
  }
  :where(carved-gallery > carved-image) { height: 100%; min-width: 0; }
`);
document.adoptedStyleSheets = [...document.adoptedStyleSheets, gallerySheet];

customElements.define('carved-image', CarvedImage);
customElements.define('carved-gallery', class CarvedGallery extends HTMLElement {});
