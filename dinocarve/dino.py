"""DINOv2-S/14 (with registers) guidance at pixel resolution: a saliency map and part features.

saliency: last-block CLS->patch attention, mean over heads, scaled by its 99th percentile. The
    register tokens (Darcet et al. 2023) absorb the high-norm artifacts that DINOv2 without
    registers puts on background patches, so the map needs no artifact filtering.
parts: block-8 patch features (zero-based, two thirds of the depth), later clustered by k-means.
Both are lifted from the 14px token grid to pixels by JAFAR. JAFAR computes attention from image
queries to token keys (keys conditioned on the last-block features) and multiplies by values, so
any token-grid map can be passed as a value and is upsampled with the same weights."""
import os
import pickle
import types
import urllib.request
from pathlib import Path

import torch
from einops import rearrange
from torch.nn import functional as F

from .jafar import JAFAR, create_coordinate

BACKBONE = 'dinov2_vits14_reg'
JAFAR_URL = 'https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_patch14_reg4_dinov2.pth'
CACHE = Path(os.environ.get('DINOCARVE_CACHE', Path.home() / '.cache' / 'dinocarve'))
PART_LAYER = 8
MEAN = torch.tensor([.485, .456, .406])[None, :, None, None]
STD = torch.tensor([.229, .224, .225])[None, :, None, None]
_models = {}


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    """JAFAR checkpoints also pickle their omegaconf training config, which we don't need.
    Stub it out and refuse every global other than tensors and builtin containers."""
    ALLOWED = {('collections', 'OrderedDict'), ('collections', 'defaultdict'), ('builtins', 'dict'), ('builtins', 'list'),
               ('builtins', 'int'), ('__builtin__', 'dict'), ('__builtin__', 'list'), ('__builtin__', 'long'),
               ('torch._utils', '_rebuild_tensor_v2'), ('torch', 'FloatStorage')}

    def find_class(self, module, name):
        if module.startswith('omegaconf') or (module, name) == ('typing', 'Any'):
            return _Stub
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'refusing {module}.{name}')


def load_jafar_state(path):
    safe = types.SimpleNamespace(__name__='pickle', Unpickler=_Unpickler, load=pickle.load)
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=safe)['jafar']


def models(device):
    """(DINOv2-S/14 with registers, JAFAR) on `device`. Downloads on first use: DINOv2 via torch.hub, JAFAR weights to CACHE."""
    if device not in _models:
        dino = torch.hub.load('facebookresearch/dinov2', BACKBONE).eval().requires_grad_(False).to(device)
        path = CACHE / Path(JAFAR_URL).name
        if not path.exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(JAFAR_URL, path)
        jafar = JAFAR(input_dim=3, qk_dim=128, v_dim=384, feature_dim=384, kernel_size=1, num_heads=4)
        jafar.load_state_dict(load_jafar_state(path))
        _models[device] = dino, jafar.eval().requires_grad_(False).to(device)
    return _models[device]


def cls_attention(dino, normed, grid):
    """[1,1,h,w] last-block CLS->patch attention in [0,1]."""
    x = dino.prepare_tokens_with_masks(normed)
    for blk in dino.blocks[:-1]:
        x = blk(x)
    blk = dino.blocks[-1]
    a = blk.attn
    B, N, C = x.shape
    qkv = a.qkv(blk.norm1(x)).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
    first_patch = 1 + dino.num_register_tokens
    att = ((qkv[0][:, :, :1] * a.scale) @ qkv[1].transpose(-2, -1)).softmax(-1)[:, :, 0, first_patch:]
    sal = att.mean(1).reshape(B, 1, *grid)
    return (sal / torch.quantile(sal.flatten(), .99).clamp_min(1e-8)).clamp(0, 1)


def jafar_upsample(jafar, normed, last, values, out, budget=2**30):
    """Upsample token-grid maps {name: [1,C,h,w]} to `out` (H,W) with JAFAR's attention.
    Query rows are independent, so they are processed in chunks of about `budget` bytes."""
    enc = jafar.image_encoder(normed)
    eh = enc.shape[2]
    coords = create_coordinate(enc.shape[-2], enc.shape[-1], device=enc.device)
    enc = rearrange(jafar.rope(rearrange(enc, 'b c h w -> b (h w) c'), coords), 'b (h w) c -> b c h w', h=eh)
    lh, lw = last.shape[-2:]
    q = jafar.norm(F.adaptive_avg_pool2d(jafar.query_encoder(enc), output_size=out))
    k = F.adaptive_avg_pool2d(jafar.key_encoder(enc), output_size=(lh, lw))
    k = jafar.sft_key(k, jafar.key_features_encoder(F.normalize(last, dim=1)))
    blk = jafar.cross_decode
    ca = blk.cross_attn
    q = rearrange(blk.conv2d(q), 'b c h w -> b (h w) c')
    k = rearrange(k, 'b c h w -> b (h w) c')
    qn, kn, vn = ca.norm_q(q), ca.norm_k(k), ca.norm_v(rearrange(last, 'b c h w -> b (h w) c'))
    vals = {n: rearrange(v, 'b c h w -> b (h w) c') for n, v in values.items()}
    parts = {n: [] for n in values}
    chunk = max(1024, budget // (kn.shape[1] * 4 * 4))
    for i in range(0, qn.shape[1], chunk):
        _, attn = ca.attention(qn[:, i:i + chunk], kn, vn, average_attn_weights=True)
        for n in values:
            parts[n].append(torch.einsum('bij,bjd->bid', attn, vals[n]))
    return {n: rearrange(torch.cat(parts[n], 1), 'b (h w) c -> b c h w', h=out[0]) for n in values}


@torch.inference_mode()
def guidance(rgba, device):
    """[H,W,4] uint8 -> (saliency [H,W] in [0,1], part features [H,W,384] unit-norm), on `device`."""
    dino, jafar = models(device)
    H, W = rgba.shape[:2]
    x = torch.as_tensor(rgba, dtype=torch.float32, device=device)[None] / 255
    rgb = (x[..., :3] * x[..., 3:] + .5 * (1 - x[..., 3:])).permute(0, 3, 1, 2)
    size = ((H + 13) // 14 * 14, (W + 13) // 14 * 14)
    rgb = F.interpolate(rgb, size=size, mode='bicubic', align_corners=False).clamp(0, 1)
    normed = (rgb - MEAN.to(device)) / STD.to(device)
    part, last = dino.get_intermediate_layers(normed, n=[PART_LAYER, 11], reshape=True, norm=True)
    sal = cls_attention(dino, normed, last.shape[-2:])
    up = jafar_upsample(jafar, normed, last, {'part': part, 'saliency': sal}, (H // 2, W // 2))
    up = {n: F.interpolate(v, size=(H, W), mode='bilinear', align_corners=False)[0] for n, v in up.items()}
    return up['saliency'][0].clamp(0, 1), F.normalize(up['part'], dim=0).permute(1, 2, 0)
