/* Quilt worker for web/index.html. Owns a subset of tiles and turns their packed seams into nested rank
   maps with a Fenwick tree per row (order statistics), then into per-level GPU data.
   Format: see "format" in quilt.json (written by python -m dinocarve.quilt). */
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
/* Optional saliency.png (any resolution, nearest-sampled to W x H), or null. */
async function saliency(url, W, H) {
  try {
    const r = await fetch(url); if (!r.ok) return null;
    const b = await createImageBitmap(await r.blob(), {premultiplyAlpha: 'none', colorSpaceConversion: 'none'}), w = b.width, h = b.height;
    const g = new OffscreenCanvas(w, h).getContext('2d', {willReadFrequently: true}); g.drawImage(b, 0, 0); b.close();
    const px = g.getImageData(0, 0, w, h).data, s = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) s[y * W + x] = px[(Math.min(h - 1, (y * h / H) | 0) * w + Math.min(w - 1, (x * w / W) | 0)) * 4];
    return s;
  } catch { return null; }
}
/* minByValue[v] = smallest rank among pixels with saliency v: the protect limit for threshold T is
   the min over v >= T * 255 (computed on the main thread, so the threshold slider needs no worker call). */
function minByValue(rank, sal) {
  const m = new Uint16Array(256).fill(65535); if (!sal) return m;
  for (let i = 0; i < rank.length; i++) { const v = sal[i]; if (rank[i] < m[v]) m[v] = rank[i]; }
  return m;
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
/* A tile loads its image and dino ('smart') seams up front. Other seam sets and the saliency map are
   fetched the first time they are needed (Vanilla, Protect): most visitors never download them. */
async function load(meta, base) {
  const t = {meta, base, sal: null, salP: null, sets: {}, setP: {}};
  [t.color] = await Promise.all([imagePixels(`${base}${meta.name}/image.webp`), seams(t, 'smart')]);
  tiles.set(meta.name, t);
}
function seams(t, e) {
  return t.setP[e] ||= fetchBuffer(`${t.base}${t.meta.name}/seams${e === 'smart' ? '' : '-' + e}.bin.gz`, true).then(buf => {
    const S = t.sets[e] = seamSet(t.meta, buf); if (t.sal) S.vMin = minByValue(S.v, t.sal); return S; })
    .catch(err => { delete t.setP[e]; throw err; });             // a failed fetch is retried, not cached
}
function loadSaliency(t) {
  return t.salP ||= saliency(`${t.base}${t.meta.name}/saliency.png`, t.meta.width, t.meta.height).then(sal => {
    t.sal = sal; for (const S of Object.values(t.sets)) S.vMin = minByValue(S.v, sal); })
    .catch(err => { t.salP = null; throw err; });
}
const vMins = t => Object.fromEntries(Object.entries(t.sets).filter(([, S]) => S.vMin).map(([e, S]) => [e, S.vMin]));
function hMinOf(t, L) {                                             // smallest height-seam rank per saliency value, for Protect
  const m = new Uint16Array(256).fill(65535), {width: W, height: H} = t.meta; if (!t.sal) return null;
  for (let y = 0, i = 0; y < H; y++) for (let x = 0; x < L.wj; x++, i++) { const v = t.sal[y * W + L.ox[i]]; if (L.hr[i] < m[v]) m[v] = L.hr[i]; }
  return m;
}

/* Image carved to widths[j] (row-major [H][wj]) and its horizontal-seam ranks (row-major). */
function computeLevel(t, j, energy) {
  const S = t.sets[energy], {width: W, height: H, max_remove_y: Ky, widths} = t.meta, wj = widths[j], k = W - wj;
  const color = new Uint32Array(wj * H), ox = new Uint16Array(wj * H), byCol = new Uint16Array(wj * H).fill(65535), hr = new Uint32Array(wj * H);
  for (let y = 0, o = 0; y < H; y++) for (let i = y * W, e = i + W; i < e; i++) if (S.v[i] >= k) { color[o] = t.color[i]; ox[o++] = i - y * W; }
  replay(S.bytes, S.offsets[j], Ky, wj, H, byCol);              // rows = x, columns = y
  for (let x = 0; x < wj; x++) for (let y = 0; y < H; y++) hr[y * wj + x] = byCol[x * H + y];
  return {wj, color, hr, ox};
}
onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {                  // in priority order; one broken tile does not stop the others
      for (const m of data.tiles) {
        try { await load(m, data.base); postMessage({type: 'loaded', name: m.name, energies: m.energies || ['smart']}); }
        catch (e) { postMessage({type: 'loadfail', name: m.name, message: e.message}); }
      }
    }
    else if (data.type === 'saliency') {         // Protect turned on: fetch saliency, report width limits
      const t = tiles.get(data.name); await loadSaliency(t); postMessage({type: 'saliency', name: data.name, vMin: vMins(t)});
    }
    else if (data.type === 'hmin') {             // height limits of a level decoded before the saliency arrived
      const t = tiles.get(data.name); await loadSaliency(t); await seams(t, data.energy);
      postMessage({type: 'hmin', name: data.name, j: data.j, energy: data.energy, hMin: hMinOf(t, computeLevel(t, data.j, data.energy))});
    }
    else if (data.type === 'glevel') {           // WebGL renderer: level colours + seam table Y[r][x] (original row of seam r in column x)
      const t = tiles.get(data.name), S = await seams(t, data.energy), {height: H, max_remove_y: Ky} = t.meta, L = computeLevel(t, data.j, data.energy), wj = L.wj;
      const Y = new Uint16Array(Ky * wj), hMin = hMinOf(t, L);
      for (let y = 0; y < H; y++) for (let x = 0, i = y * wj; x < wj; x++, i++) {
        const r = L.hr[i]; if (r < Ky) Y[r * wj + x] = y;
      }
      // Width seams between this level and the next: every row holds exactly one pixel of each, as
      // (x in this level, rank offset). Rendering gives them fractional widths: continuous width.
      const W = t.meta.width, widths = t.meta.widths, j = data.j;
      const wj1 = j + 1 < widths.length ? widths[j + 1] : wj, n = wj - wj1, k = W - wj, k1 = W - wj1;
      const marks = new Uint16Array(Math.max(n, 1) * H * 2);
      for (let y = 0; y < H && n > 0; y++) {
        let cnt = 0, x = 0;
        for (let i = y * W, e = i + W; i < e; i++) { const rv = S.v[i]; if (rv < k) continue; if (rv < k1) { const o = (y * n + cnt++) * 2; marks[o] = x; marks[o + 1] = rv - k; } x++; }
        if (cnt !== n) throw Error(`width seams: row ${y} has ${cnt}, expected ${n}`);
      }
      postMessage({type: 'glevel', name: data.name, j, energy: data.energy, wj, n, color: L.color, seams: Y, marks, hMin, vMin: S.vMin}, [L.color.buffer, Y.buffer, marks.buffer]);
    }
  } catch (e) { postMessage({type: 'error', message: e.message, req: data.type, name: data.name, j: data.j, energy: data.energy}); }
};
