"""Runnable timing and oautput-shape probe for issue #1126 (ActivationCache repr).

Uses the repository's real ``ActivationCache`` class to separate:
1. ``ActivationCache`` formatting cost vs. cached tensor volume (batch / seq / d_model).
2. ``ActivationCache.__repr__`` scaling with hook key count (including Bridge aliases).
3. Tensor-formatting paths (raw ``dict`` / ``return_cache_object=False`` / ``(logits, cache)``).

Usage:
    uv run python debugging/activation_cache_repr_probe.py
    uv run python debugging/activation_cache_repr_probe.py --device cuda
    uv run python debugging/activation_cache_repr_probe.py --model tiny-stories-1M
    uv run python debugging/activation_cache_repr_probe.py --model gpt2 --bridge --compat
"""

import argparse
import importlib
import inspect
import pprint
import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch

from transformer_lens.ActivationCache import ActivationCache


def _ipython_pretty() -> Callable[[Any], str] | None:
    try:
        fn = getattr(importlib.import_module("IPython.lib.pretty"), "pretty", None)
    except ImportError:
        return None
    return cast(Callable[[Any], str], fn) if callable(fn) else None


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def _measure(fn: Callable[[], str], dev: torch.device, repeats: int) -> tuple[float, int, int]:
    times: list[float] = []
    out = ""
    for _ in range(max(1, repeats)):
        _sync(dev)
        t0 = time.perf_counter()
        out = fn()
        _sync(dev)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), len(out), (out.count("\n") + 1 if out else 0)


def build_cache(
    n_layers: int,
    b: int,
    s: int,
    d: int,
    h: int,
    dev: torch.device,
    aliases: bool = False,
) -> ActivationCache:
    dh, d_mlp = max(1, d // max(1, h)), d * 4
    cd: dict[str, torch.Tensor] = {
        "hook_embed": torch.randn(b, s, d, device=dev),
        "hook_pos_embed": torch.randn(b, s, d, device=dev),
    }
    for i in range(n_layers):
        p = f"blocks.{i}"
        for k in ("hook_resid_pre", "hook_resid_mid", "hook_resid_post", "attn.hook_out"):
            cd[f"{p}.{k}"] = torch.randn(b, s, d, device=dev)
        for ln in ("ln1", "ln2"):
            cd[f"{p}.{ln}.hook_scale"] = torch.ones(b, s, 1, device=dev)
            cd[f"{p}.{ln}.hook_normalized"] = torch.randn(b, s, d, device=dev)
        for qkv in ("hook_q", "hook_k", "hook_v", "hook_z"):
            cd[f"{p}.attn.{qkv}"] = torch.randn(b, s, h, dh, device=dev)
        for pat in ("hook_attn_scores", "hook_pattern"):
            cd[f"{p}.attn.{pat}"] = torch.randn(b, h, s, s, device=dev)
        cd[f"{p}.mlp.hook_pre"] = torch.randn(b, s, d_mlp, device=dev)
        cd[f"{p}.mlp.hook_post"] = torch.randn(b, s, d_mlp, device=dev)
        cd[f"{p}.hook_mlp_out"] = torch.randn(b, s, d, device=dev)
        if aliases:
            for alias, src in [
                ("hook_in", "hook_resid_pre"),
                ("hook_out", "hook_resid_post"),
                ("attn.q.hook_out", "attn.hook_q"),
                ("attn.k.hook_out", "attn.hook_k"),
                ("attn.v.hook_out", "attn.hook_v"),
                ("mlp.hook_in", "ln2.hook_normalized"),
                ("mlp.hook_out", "hook_mlp_out"),
            ]:
                cd[f"{p}.{alias}"] = cd[f"{p}.{src}"]
    cd["ln_final.hook_scale"] = torch.ones(b, s, 1, device=dev)
    cd["ln_final.hook_normalized"] = torch.randn(b, s, d, device=dev)
    _sync(dev)
    return ActivationCache(cd, model=None)


def _mb(cache: ActivationCache) -> float:
    return sum(t.numel() * t.element_size() for t in cache.cache_dict.values()) / (1024.0 * 1024.0)


def _hdr(title: str) -> None:
    cols = f"{'Path':<44} | {'Keys':>5} | {'Cache MB':>8} | {'Median ms':>9} | {'Chars':>10}"
    print(f"\n=== {title} ===\n{cols} | {'Lines':>7}\n" + "-" * 98)


def _row(label: str, keys: int, mb: float, stats: tuple[float, int, int]) -> None:
    ms, chars, lines = stats
    print(f"{label:<44} | {keys:>5} | {mb:>8.2f} | {ms:>9.3f} | {chars:>10} | {lines:>7}")


def run_synthetic(dev: torch.device, repeats: int, sample: int) -> None:
    ipy = _ipython_pretty()
    _hdr("1. Tensor-Volume Invariance on ActivationCache (Fixed 12 Layers)")
    for tag, b, s, d, h in [("tiny [1,8,64]", 1, 8, 64, 4), ("large [2,256,768]", 2, 256, 768, 12)]:
        c = build_cache(12, b, s, d, h, dev)
        mb = _mb(c)
        fns: list[tuple[str, Callable[[], str]]] = [
            (f"repr(cache) [{tag}]", partial(repr, c)),
            (f"str(cache) [{tag}]", partial(str, c)),
            (f"pprint.pformat(cache) [{tag}]", partial(pprint.pformat, c)),
        ]
        if ipy is not None:
            fns.append((f"IPython.pretty(cache) [{tag}]", partial(ipy, c)))
        for name, fn in fns:
            _row(name, len(c), mb, _measure(fn, dev, repeats))

    _hdr("2. Key-Count Scaling & Unwrapped Single-Line Width")
    cases = [("12L", 12, False), ("32L", 32, False), ("80L", 80, False), ("80L+aliases", 80, True)]
    for tag, nl, al in cases:
        c = build_cache(nl, 1, 2, 8, 2, dev, aliases=al)
        _row(f"repr(cache) [{tag}]", len(c), _mb(c), _measure(partial(repr, c), dev, repeats))

    _hdr("3. ActivationCache vs. Tensor-Formatting Paths (12L, [2, 128, 512])")
    c = build_cache(12, 2, 128, 512, 8, dev)
    logits = torch.randn(2, 128, 50257, device=dev)
    mb = _mb(c)
    _row("repr(cache) [ActivationCache]", len(c), mb, _measure(partial(repr, c), dev, repeats))
    _row("repr((logits, cache)) [tuple]", len(c), mb, _measure(partial(repr, (logits, c)), dev, 1))
    items = list(c.cache_dict.items())
    sub = dict(items if sample <= 0 else items[:sample])
    lbl = "all" if sample <= 0 else f"first {len(sub)}"
    _row(f"repr(raw_dict) [{lbl} keys]", len(sub), mb, _measure(partial(repr, sub), dev, 1))
    if ipy is not None:
        _row(f"IPython.pretty(raw_dict) [{lbl}]", len(sub), mb, _measure(partial(ipy, sub), dev, 1))


def run_live(args: argparse.Namespace, dev: torch.device) -> None:
    _hdr(f"4. Live run_with_cache Probe ({args.model}, bridge={args.bridge})")
    if args.bridge:
        from transformer_lens.model_bridge import TransformerBridge

        model: Any = TransformerBridge.boot_transformers(args.model, device=dev)
        if args.compat:
            model.enable_compatibility_mode()
    else:
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(args.model, device=str(dev))
    _sync(dev)
    t0 = time.perf_counter()
    logits, cache = model.run_with_cache(args.prompt)
    _sync(dev)
    print(f"Forward + run_with_cache wall time: {(time.perf_counter() - t0) * 1000.0:.3f} ms")
    assert isinstance(cache, ActivationCache)
    mb = _mb(cache)
    _row("repr(cache) [live]", len(cache), mb, _measure(partial(repr, cache), dev, args.repeats))
    tup_stats = _measure(partial(repr, (logits, cache)), dev, 1)
    _row("repr((logits, cache)) [live tuple]", len(cache), mb, tup_stats)
    items = list(cache.cache_dict.items())
    sub = dict(items if args.dict_sample_keys <= 0 else items[: args.dict_sample_keys])
    _row(f"repr(cache_dict) [n={len(sub)}]", len(sub), mb, _measure(partial(repr, sub), dev, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe ActivationCache.__repr__ paths for #1126.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dict-sample-keys", type=int, default=24)
    ap.add_argument("--model", default=None)
    ap.add_argument("--bridge", action="store_true")
    ap.add_argument("--compat", action="store_true")
    ap.add_argument("--prompt", default="When John and Mary went to the store, Mary gave milk to")
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"Loaded ActivationCache from: {inspect.getfile(ActivationCache)}")
    print(f"Target device: {dev} | IPython pretty available: {_ipython_pretty() is not None}")
    run_synthetic(dev, args.repeats, args.dict_sample_keys)
    if args.model is not None:
        run_live(args, dev)


if __name__ == "__main__":
  main()
    main()
