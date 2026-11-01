/* Quilt carving worker. Owns a subset of tiles; turns packed seams into nested rank maps with a
   Fenwick tree per row (order statistics) and renders carved states on request as ImageBitmaps.
   Format: see "format" in dist/quilt/quilt.json. */
const tiles = new Map();

async function fetchBuffer(url, gz) {
  const r = await fetch(url); if (!r.ok) throw Error(`${url}: ${r.status}`);
  const buf = await r.arrayBuffer(), b = new Uint8Array(buf, 0, 2);
  if (!gz || b[0] !== 0x1f || b[1] !== 0x8b) return buf;            // some hosts already decoded it (Content-Encoding: gzip)
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
async function imagePixels(url) {
  const r = await fetch(url); if (!r.ok) throw Error(`${url}: ${r.status}`);
  const b = await createImageBitmap(await r.blob(), {premultiplyAlpha: 'none', colorSpaceConversion: 'none'});
  const c = new OffscreenCanvas(b.width, b.height), g = c.getContext('2d', {willReadFrequently: true});
  g.drawImage(b, 0, 0); return new Uint32Array(g.getImageData(0, 0, b.width, b.height).data.buffer);
}

/* Replay K seams over R rows of a C-wide image into rank[r * C + original column]. */
function replay(bytes, offset, K, R, C, rank) {
  const n = (R - 1 + 3) >> 2, size = 2 + n, cur = new Int32Array(K), tree = new Int32Array(C + 1);
  let LOG = 1; while (LOG * 2 <= C) LOG *= 2;
  for (let k = 0; k < K; k++) cur[k] = bytes[offset + k * size] | bytes[offset + k * size + 1] << 8;
  for (let r = 0; r < R; r++) {
    if (r) for (let k = 0; k < K; k++) {
      const code = bytes[offset + k * size + 2 + ((r - 1) >> 2)] >> (((r - 1) & 3) * 2) & 3;
      cur[k] += code === 1 ? 1 : code === 2 ? -1 : 0;
    }
    for (let i = 1; i <= C; i++) tree[i] = i & -i;             // Fenwick of all-ones
    const row = r * C;
    for (let k = 0; k < K; k++) {
      let pos = 0, rem = cur[k] + 1;                             // (cur+1)-th alive column
      for (let step = LOG; step; step >>= 1) { const nx = pos + step; if (nx <= C && tree[nx] < rem) { pos = nx; rem -= tree[nx]; } }
      rank[row + pos] = k;
      for (let i = pos + 1; i <= C; i += i & -i) tree[i]--;
    }
  }
  return offset + K * size;
}

/* A seam set per energy ('smart' = ours, 'textbook' = vanilla): vertical ranks + packed level seams. */
function seamSet(meta, buf) {
  const bytes = new Uint8Array(buf), {width: W, height: H, max_remove_x: Kx, max_remove_y: Ky, widths} = meta;
  const v = new Uint16Array(W * H).fill(65535);
  let off = replay(bytes, 0, Kx, H, W, v); const offsets = [];
  for (const wj of widths) { offsets.push(off); off += Ky * (2 + ((wj - 1 + 3) >> 2)); }
  return {v, bytes, offsets};
}
async function load(meta, base, energies = ['smart']) {
  const file = e => `${base}${meta.name}/seams${e === 'smart' ? '' : '-' + e}.bin.gz`;
  const [color, ...bufs] = await Promise.all([imagePixels(`${base}${meta.name}/image.webp`), ...energies.map(e => fetchBuffer(file(e), true).catch(() => null))]);
  const sets = {}; energies.forEach((e, i) => { if (bufs[i]) sets[e] = seamSet(meta, bufs[i]); });
  tiles.set(meta.name, {meta, color, sets, levels: new Map()});
}

const MAX_LEVELS = 4;                      // decoded width levels kept per tile (LRU, 2D renderer)
/* Image carved to widths[j] (row-major [H][wj]) and its horizontal-seam ranks (row-major). */
function computeLevel(t, j, energy = 'smart') {
  const S = t.sets[energy] || t.sets.smart, {width: W, height: H, max_remove_y: Ky, widths} = t.meta, wj = widths[j], k = W - wj;
  const color = new Uint32Array(wj * H), byCol = new Uint16Array(wj * H).fill(65535), hr = new Uint32Array(wj * H);
  for (let y = 0, o = 0; y < H; y++) for (let i = y * W, e = i + W; i < e; i++) if (S.v[i] >= k) color[o++] = t.color[i];
  replay(S.bytes, S.offsets[j], Ky, wj, H, byCol);              // rows = x, columns = y
  for (let x = 0; x < wj; x++) for (let y = 0; y < H; y++) hr[y * wj + x] = byCol[x * H + y];
  return {wj, color, hr};
}
function level(t, j) {
  let L = t.levels.get(j); if (L) { t.levels.delete(j); t.levels.set(j, L); return L; }
  if (t.levels.size >= MAX_LEVELS) t.levels.delete(t.levels.keys().next().value);
  L = computeLevel(t, j); t.levels.set(j, L); return L;
}

function render(t, j, ky) {
  const {height: H} = t.meta, {wj, color, hr} = level(t, j), oh = H - ky, out = new Uint32Array(wj * oh), cnt = new Uint16Array(wj);
  for (let y = 0, i = 0; y < H; y++) for (let x = 0; x < wj; x++, i++) if (hr[i] >= ky) out[cnt[x]++ * wj + x] = color[i];
  const c = new OffscreenCanvas(wj, oh); c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(out.buffer), wj, oh), 0, 0);
  return c.transferToImageBitmap();
}

onmessage = async ({data}) => {
  try {
    if (data.type === 'load') { for (const m of data.tiles) { await load(m, data.base, m.energies && m.energies.length ? m.energies : data.energies); postMessage({type: 'loaded', name: m.name, energies: Object.keys(tiles.get(m.name).sets)}); } }   // in priority order
    else if (data.type === 'level') {            // GPU renderer: raw level data, transferred
      const t = tiles.get(data.name), L = computeLevel(t, data.j);
      postMessage({type: 'level', name: data.name, j: data.j, wj: L.wj, color: L.color, hr: L.hr}, [L.color.buffer, L.hr.buffer]);
    }
    else if (data.type === 'glevel') {           // WebGL renderer: level colours + seam table Y[r][x] (original row of seam r in column x)
      const t = tiles.get(data.name), {height: H, max_remove_y: Ky} = t.meta, L = computeLevel(t, data.j, data.energy), wj = L.wj;
      const Y = new Uint16Array(Ky * wj);
      for (let y = 0; y < H; y++) for (let x = 0, i = y * wj; x < wj; x++, i++) { const r = L.hr[i]; if (r < Ky) Y[r * wj + x] = y; }
      // Width seams between this level and the next: every row holds exactly one pixel of each, as
      // (x in this level, rank offset). Rendering gives them fractional widths: continuous width.
      const S = t.sets[data.energy] || t.sets.smart, W = t.meta.width, widths = t.meta.widths, j = data.j;
      const wj1 = j + 1 < widths.length ? widths[j + 1] : wj, n = wj - wj1, k = W - wj, k1 = W - wj1;
      const marks = new Uint16Array(Math.max(n, 1) * H * 2);
      for (let y = 0; y < H && n > 0; y++) {
        let cnt = 0, x = 0;
        for (let i = y * W, e = i + W; i < e; i++) { const rv = S.v[i]; if (rv < k) continue; if (rv < k1) { const o = (y * n + cnt++) * 2; marks[o] = x; marks[o + 1] = rv - k; } x++; }
        if (cnt !== n) throw Error(`width seams: row ${y} has ${cnt}, expected ${n}`);
      }
      postMessage({type: 'glevel', name: data.name, j, energy: data.energy || 'smart', wj, n, color: L.color, seams: Y, marks}, [L.color.buffer, Y.buffer, marks.buffer]);
    }
    else if (data.type === 'render') {
      const t = tiles.get(data.name), bitmap = render(t, data.j, data.ky);
      postMessage({type: 'state', name: data.name, j: data.j, ky: data.ky, bitmap}, [bitmap]);
    }
  } catch (e) { postMessage({type: 'error', message: e.message, name: data.name}); }
};
